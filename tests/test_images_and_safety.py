import threading
import time
from typing import ClassVar
from unittest import mock

import pytest
from huggingface_hub.errors import HfHubHTTPError
from PIL import Image

import app.multimodal as mm
from app.safety import check_prompt

LILA = ("Lila", "young girl, about twelve, curly brown hair")
NIGHT = ("Night", "black cat, glossy fur, green eyes")


# === Image prompts ===

def test_names_are_replaced_by_descriptions():
    # Image models don't know names: they draw extra people or write the name on the picture.
    prompt = mm.build_image_prompt("Lila shows Night the biscuits.", "Fantasy", "Hopeful",
                                   characters=[LILA, NIGHT])
    assert "Lila" not in prompt and "Night" not in prompt
    assert ("a young girl (about twelve, curly brown hair) shows a black cat "
            "(glossy fur, green eyes) the biscuits.") in prompt


def test_later_mentions_and_possessives_use_the_short_form():
    prompt = mm.build_image_prompt("Lila smiles while Lila's apron glows.", "Fantasy", "Hopeful",
                                   characters=[LILA])
    assert "while the young girl's apron glows" in prompt


def test_characters_missing_from_the_scene_are_still_described():
    prompt = mm.build_image_prompt("A storm rolls in.", "Fantasy", "Hopeful", characters=[("Rook", "old sailor")])
    assert "Also in the scene: an old sailor." in prompt


def style_parts(style):
    return mm.ART_STYLES[style].split(", ")


def test_style_genre_and_mood_are_applied():
    prompt = mm.build_image_prompt("A storm rolls in.", "Fantasy", "Eerie", style="Pixel art")
    assert all(part in prompt for part in style_parts("Pixel art"))
    assert prompt.endswith("fantasy story, eerie atmosphere")


def test_core_style_comes_first_because_sd_reads_only_77_tokens():
    prompt = mm.build_image_prompt("A storm rolls in.", "Fantasy", "Eerie", style="Watercolor")
    assert prompt.startswith(style_parts("Watercolor")[0] + ". A storm rolls in.")


def test_unknown_style_uses_the_default():
    prompt = mm.build_image_prompt("A storm.", "Fantasy", "Eerie", style="Not a style")
    assert prompt.startswith(style_parts(mm.DEFAULT_ART_STYLE)[0])


# === Matching names in the scene to the cast ===

MARA = ("Mara Lorn", "young cartographer, cropped brown hair, leather coat, scar above left eyebrow")
VESH = ("Keeper Vesh", "guardian spirit, midnight blue cloak, glowing crescent")


def test_first_name_alone_still_gets_the_description():
    # The live bug: the cast said "Mara Lorn", the scene said "Mara", so she was drawn as a stranger.
    prompt = mm.build_image_prompt("Mara holds a map.", "Adventure", "Epic", characters=[MARA])
    assert "Mara" not in prompt
    assert "a young cartographer (cropped brown hair, leather coat, scar above left eyebrow) holds a map." in prompt
    assert "Also in the scene" not in prompt


def test_titles_are_ignored_when_matching():
    prompt = mm.build_image_prompt("Vesh raises a lantern.", "Fantasy", "Eerie", characters=[VESH])
    assert "a guardian spirit (midnight blue cloak, glowing crescent) raises a lantern." in prompt
    # A title on its own could be anyone, so it never matches.
    assert mm.match_cast_name("the Keeper", ["Keeper Vesh"]) is None


def test_only_whole_words_match():
    prompt = mm.build_image_prompt("Annie waves at the planner.", "Drama", "Cozy", characters=[("Ann", "old woman")])
    assert prompt.count("Annie") == 1 and "Also in the scene: an old woman." in prompt


def test_the_longest_matching_name_wins():
    cast = [("Mara", "young girl"), ("Mara Lorn", "old cartographer")]
    prompt = mm.build_image_prompt("Mara Lorn hugs Mara.", "Drama", "Cozy", characters=cast)
    assert "an old cartographer hugs a young girl." in prompt


def test_a_word_shared_by_two_names_matches_neither():
    cast = {"Mara Lorn": "young cartographer", "Ellis Lorn": "old sailor"}
    assert mm.match_cast_name("Lorn", list(cast)) is None
    assert mm.match_cast_name("Mara", list(cast)) == "Mara Lorn"
    assert mm.match_cast_name("Captain Ellis", list(cast)) == "Ellis Lorn"
    prompt = mm.build_image_prompt("Ellis shouts at Lorn.", "Drama", "Tense", characters=list(cast.items()))
    assert "an old sailor shouts at Lorn." in prompt


def test_scene_cast_uses_exact_names_and_falls_back_to_partial_ones():
    cast = dict([MARA, VESH, ("Night", "black cat")])
    # Mentioned characters come first, in order; listed-but-unmentioned ones follow; unknowns are dropped.
    found = mm.scene_cast("Vesh bows to Mara.", ["Mara Lorn", "Night", "Stranger"], cast)
    assert [name for name, _ in found] == ["Keeper Vesh", "Mara Lorn", "Night"]


# === Fitting Stable Diffusion's 77-token window ===

def words(text):
    return len(text.split())


CAST = [MARA, VESH, ("Night", "black cat, sleek black fur, red collar, one white paw")]
SCENE = "Mara hands Vesh a glowing map while Night watches."


def test_without_a_token_counter_nothing_is_dropped():
    prompt = mm.build_image_prompt(SCENE, "Adventure", "Epic", "Watercolor", CAST)
    assert prompt.endswith("adventure story, epic atmosphere") and "one white paw" in prompt


def test_a_long_prompt_is_shrunk_until_it_fits():
    full = mm.build_image_prompt(SCENE, "Adventure", "Epic", "Watercolor", CAST)
    prompt = mm.build_image_prompt(SCENE, "Adventure", "Epic", "Watercolor", CAST, count_tokens=words, max_tokens=40)
    assert words(full) > 40 >= words(prompt)
    # The core style, the action and who is in it always survive.
    assert prompt.startswith("loose watercolor illustration. a young cartographer (cropped brown hair")
    assert "hands a guardian spirit" in prompt and "watches." in prompt
    assert "Mara" not in prompt and "Vesh" not in prompt and "Night" not in prompt


def test_least_important_parts_go_first():
    full = mm.build_image_prompt(SCENE, "Adventure", "Epic", "Watercolor", CAST)
    just_over = words(full) - 1
    prompt = mm.build_image_prompt(SCENE, "Adventure", "Epic", "Watercolor", CAST, count_tokens=words,
                                   max_tokens=just_over)
    # Dropping genre and mood was enough: every character detail is still there.
    assert "epic atmosphere" not in prompt and "one white paw" in prompt and "soft color washes" in prompt


def test_sd_turbo_prompts_are_counted_with_the_models_tokenizer(sd_turbo, monkeypatch):
    seen = []
    monkeypatch.setattr(mm, "_generate_in_process", lambda prompt: seen.append(prompt) or Image.new("RGB", (8, 8)))
    long_scene = " ".join(["Mara walks through the endless rain."] * 20)
    mm.generate_image(long_scene, "Drama", "Melancholic", characters=[MARA])
    assert "melancholic atmosphere" not in seen[0]  # shrinking kicked in


# === Image generation via the hosted API ===

class FakeImageClient:
    calls: ClassVar[list] = []

    def __init__(self, provider=None, api_key=None):
        pass

    def text_to_image(self, prompt, model, width, height):
        FakeImageClient.calls.append((model, width, height))
        if model == mm.IMAGE_MODEL:
            raise HfHubHTTPError("gone", response=mock.Mock(status_code=404, headers={}))
        return Image.new("RGB", (width, height))


def test_images_are_3_by_2_and_fall_back_to_the_backup_model(output_dir):
    FakeImageClient.calls = []
    with mock.patch("huggingface_hub.InferenceClient", FakeImageClient):
        path = mm.generate_image("A storm.", "Fantasy", "Eerie")
    assert [model for model, _, _ in FakeImageClient.calls] == [mm.IMAGE_MODEL, mm.IMAGE_FALLBACK_MODELS[0]]
    assert Image.open(path).size == (1152, 768)


# === Image generation in this process (SD-Turbo on a CPU Space) ===

class FakePipeline:
    """Stands in for a diffusers pipeline; records how often it ran at the same time."""

    def __init__(self):
        self.calls, self.running, self.most_at_once = [], 0, 0
        # One token per word, plus the start and end markers a real CLIP tokenizer adds.
        self.tokenizer = lambda text, **kwargs: mock.Mock(input_ids=[0, *text.split(), 0])

    def __call__(self, prompt, width, height, **settings):
        self.running += 1
        self.most_at_once = max(self.most_at_once, self.running)
        time.sleep(0.05)
        self.calls.append(settings)
        self.running -= 1
        return mock.Mock(images=[Image.new("RGB", (width, height))])


@pytest.fixture
def sd_turbo(monkeypatch, output_dir):
    pipe = FakePipeline()
    monkeypatch.setattr(mm, "IMAGE_PROVIDER", "sd-turbo")
    monkeypatch.setattr(mm, "load_image_pipeline", lambda: pipe)
    return pipe


def test_sd_turbo_draws_in_one_step_without_guidance(sd_turbo):
    path = mm.generate_image("A storm.", "Fantasy", "Eerie")
    assert sd_turbo.calls == [{"num_inference_steps": 1, "guidance_scale": 0.0}]
    assert Image.open(path).size == (768, 512)


def test_image_model_loads_once_when_two_callers_overlap(monkeypatch):
    # The startup preload and the first request asking at the same moment must share one load.
    import diffusers

    loads = []

    class SlowPipeline:
        def __init__(self):
            loads.append(self)
            time.sleep(0.2)  # a real load takes seconds, so the second caller arrives mid-load

        to = enable_attention_slicing = set_progress_bar_config = lambda self, *a, **k: self

    monkeypatch.setattr(diffusers.AutoPipelineForText2Image, "from_pretrained", lambda *a, **k: SlowPipeline())
    mm._load_image_pipeline.cache_clear()
    start = threading.Barrier(2)
    results = []

    def caller():
        start.wait()
        results.append(mm.load_image_pipeline())

    threads = [threading.Thread(target=caller) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    mm._load_image_pipeline.cache_clear()  # don't leave the fake behind for other tests
    assert len(loads) == 1
    assert len(results) == 2 and results[0] is results[1] is loads[0]


def test_in_process_pictures_are_drawn_one_at_a_time(sd_turbo):
    threads = [threading.Thread(target=mm.generate_image, args=("A storm.", "Fantasy", "Eerie")) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(sd_turbo.calls) == 3
    assert sd_turbo.most_at_once == 1


# === Safety ===

@pytest.mark.parametrize("prompt", ["write a sex scene", "a nude painting of a kid", "some nsfw story"])
def test_unacceptable_prompts_are_refused(prompt):
    assert check_prompt(prompt)


@pytest.mark.parametrize("prompt", [
    "The kids play hide and seek in the forest",
    "A knight draws a naked blade",
    "An explicit warning is painted on the castle gate",
    "A horror story in an abandoned station",
])
def test_ordinary_prompts_are_allowed(prompt):
    assert check_prompt(prompt) is None
