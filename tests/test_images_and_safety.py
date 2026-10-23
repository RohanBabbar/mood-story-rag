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
