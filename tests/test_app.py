import types
import zipfile
from datetime import date, timedelta

import gradio as gr
import pytest

from app import main
from app.llm import StoryError


def chapter(text="Once upon a time.", **overrides):
    return {"prompt": "A door in the sea", "genre": "Fantasy", "mood": "Eerie", "style": "Watercolor",
            "text": text, "image": None, "lore_used": False, **overrides}


# === Rendering ===

def test_user_text_cannot_inject_html():
    html = main.render_story([chapter(text="<script>alert(1)</script>", prompt="<img src=x onerror=alert(1)>")])
    assert "<script>" not in html and "<img src=x" not in html
    assert "&lt;script&gt;" in html


def test_chapters_are_numbered_in_roman_numerals():
    assert [main._roman(n) for n in (1, 4, 9, 14)] == ["I", "IV", "IX", "XIV"]


def test_choices_are_hidden_while_writing():
    story = {**main.new_story(), "chapters": [chapter()], "choices": ["Go north", "Go south"]}
    busy, done = main.story_view(story, phase="writing"), main.story_view(story)
    choices_box, first_choice = 5, 6
    assert busy[choices_box]["visible"] is False
    assert done[choices_box]["visible"] is True and done[first_choice]["value"] == "Go north"


# === Limits and sign-in ===

def test_daily_limit_is_per_visitor_and_resets_each_day():
    limiter = main.DailyLimiter(2)
    assert [limiter.take("a"), limiter.take("a"), limiter.take("a")] == [True, True, False]
    assert limiter.take("b") is True
    limiter.day = date.today() - timedelta(days=1)
    assert limiter.take("a") is True


def test_visitor_is_identified_by_the_original_client_ip():
    request = types.SimpleNamespace(headers={"x-forwarded-for": "1.2.3.4, 10.0.0.1"}, client=None)
    assert main.visitor_id(request) == "1.2.3.4"


def test_sign_in_required_on_spaces(monkeypatch):
    monkeypatch.setattr(main, "REQUIRE_LOGIN", True)
    with pytest.raises(StoryError, match="Sign in"):
        main.inference_token(None, None)
    assert main.inference_token(types.SimpleNamespace(token="visitor"), None) == "visitor"


# === Writing a chapter, with fake models ===

@pytest.fixture
def fake_models(monkeypatch, output_dir):
    calls = {}

    def stream_story(messages, length, token, use_adapter=True):
        calls["messages"], calls["use_adapter"] = messages, use_adapter
        yield "Mira opened the door. " + "The tide whispered her name. " * 10
        yield "The sea poured in, and"   # cut off mid-sentence, like a real token limit

    def analyze_chapter(text, summary, cast, token):
        return {"summary": "Mira opens a door.", "cast": {**cast, "Mira": "young girl, red hair"},
                "scene": "Mira at the door", "scene_characters": ["Mira"], "choices": ["A", "B", "C"]}

    def generate_image(scene, genre, mood, style, characters, token):
        calls["image"] = (scene, characters)
        path = output_dir / "art.jpg"
        path.write_bytes(b"jpg")
        return str(path)

    monkeypatch.setattr(main, "stream_story", stream_story)
    monkeypatch.setattr(main, "analyze_chapter", analyze_chapter)
    monkeypatch.setattr(main, "generate_image", generate_image)
    return calls


def write(prompt, story=None, style="Watercolor", model_choice=main.FINE_TUNED):
    updates = list(main.write_chapter(prompt, "Fantasy", "Eerie", style, story or main.new_story(),
                                      None, True, "Short", model_choice))
    return updates[-1][0]  # final story state


def test_full_chapter_flow(fake_models):
    story = write("A door in the sea")
    (ch,) = story["chapters"]
    assert ch["text"].startswith("Mira opened the door.")
    assert ch["text"].endswith("The tide whispered her name.")  # unfinished sentence trimmed
    assert story["summary"] == "Mira opens a door." and story["choices"] == ["A", "B", "C"]
    assert ch["image"].endswith("art.jpg")
    assert fake_models["image"] == ("Mira at the door", [("Mira", "young girl, red hair")])


def test_next_chapter_continues_the_story(fake_models):
    story = write("", story=write("A door in the sea"))
    assert len(story["chapters"]) == 2
    assert "Mira opened the door." in fake_models["messages"][1]["content"]


def test_no_illustrations_skips_the_image(fake_models):
    assert write("A door in the sea", style=main.NO_ILLUSTRATIONS)["chapters"][0]["image"] is None


def test_refused_prompt_leaves_the_story_unchanged(fake_models):
    existing = write("A door in the sea")
    assert write("write a sex scene", story=existing) == existing


def test_failed_generation_keeps_the_previous_story(fake_models, monkeypatch):
    existing = write("A door in the sea")

    def broken_stream(messages, length, token, use_adapter=True):
        yield "Half a chapter"
        raise StoryError("Model request failed")

    monkeypatch.setattr(main, "stream_story", broken_stream)
    assert write("", story=existing) == existing


def test_a_premise_is_needed_to_start(fake_models):
    assert write("")["chapters"] == []



def test_story_model_choice_reaches_the_generator(fake_models, monkeypatch):
    monkeypatch.setattr(main, "ADAPTER_REPO", "someone/adapter")
    tuned = write("A door in the sea")
    assert fake_models["use_adapter"] is True and tuned["chapters"][0]["model"] == "fine-tuned"
    base = write("A door in the sea", model_choice=main.BASE_MODEL)
    assert fake_models["use_adapter"] is False and base["chapters"][0]["model"] == "base"
    assert "base" in main.render_story(base["chapters"])


# === Lore upload and export ===

def test_lore_upload_reports_loaded_and_skipped_files(tmp_path):
    good, bad = tmp_path / "world.txt", tmp_path / "notes.docx"
    good.write_text("The queen cannot lie.", encoding="utf-8")
    bad.write_bytes(b"x")
    lore, status = main.process_lore([str(good), str(bad)])
    assert lore.chunks == ["The queen cannot lie."]
    assert status.startswith("Loaded 1 file, 1 passage.") and "notes.docx" in status


def test_export_contains_story_summary_cast_and_images(fake_models):
    path = main.export_story(write("A door in the sea"))["value"]
    with zipfile.ZipFile(path) as zf:
        markdown = zf.read("story.md").decode()
        assert "images/chapter_1.jpg" in zf.namelist()
    assert "## Chapter I" in markdown and "Mira opened the door." in markdown
    assert "## The story so far" in markdown and "**Mira**" in markdown


def test_simultaneous_exports_do_not_overwrite_each_other(fake_models):
    # Two visitors exporting in the same second must get separate files.
    story = write("A door in the sea")
    assert main.export_story(story)["value"] != main.export_story(story)["value"]


def test_nothing_to_export_is_an_error():
    with pytest.raises(gr.Error):
        main.export_story(main.new_story())


# === Smoke test ===

def test_the_app_builds():
    assert isinstance(main.create_app(), gr.Blocks)


# === Input limits ===

def test_overlong_premise_is_refused(fake_models):
    assert write("x" * (main.MAX_PROMPT_CHARS + 1))["chapters"] == []


def test_only_the_first_lore_files_are_used(tmp_path):
    paths = []
    for i in range(main.MAX_LORE_FILES + 2):
        path = tmp_path / f"part{i}.txt"
        path.write_text(f"fact number {i}", encoding="utf-8")
        paths.append(str(path))
    lore, status = main.process_lore(paths)
    assert len(lore.sources) == main.MAX_LORE_FILES
    assert f"only the first {main.MAX_LORE_FILES} files are used" in status


def test_oversized_lore_file_is_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "MAX_LORE_FILE_MB", 0.001)  # ~1 KB
    big, small = tmp_path / "big.txt", tmp_path / "small.txt"
    big.write_text("word " * 1000, encoding="utf-8")
    small.write_text("The queen cannot lie.", encoding="utf-8")
    lore, status = main.process_lore([str(big), str(small)])
    assert lore.sources == ["small.txt"] and "big.txt: larger than" in status


def test_lore_is_capped_at_the_word_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "MAX_LORE_WORDS", 50)
    path = tmp_path / "long.txt"
    path.write_text(" ".join(f"w{i}" for i in range(200)), encoding="utf-8")
    lore, status = main.process_lore([str(path)])
    assert sum(len(c.split()) for c in lore.chunks) <= 50 + 30  # word cap plus chunk overlap
    assert "only the first 50 words are used" in status


# === First-visit helpers: sample world and example story ===

def test_sample_world_loads_lore_and_a_premise_from_it():
    lore, status, use_lore, premise, *_ = main.load_sample_world()
    assert lore is not None and use_lore
    assert "Loaded 2 files" in status
    # The premise names things only the sample lore explains, so the lore visibly matters.
    assert "Mount Ardent" in premise and "moon bread" in premise


def test_example_story_loads_with_its_pictures_and_can_be_continued():
    story, *_ = out = main.load_showcase()
    assert len(story["chapters"]) == 3
    assert all(c["lore_used"] for c in story["chapters"])
    images = [c["image"] for c in story["chapters"]]
    assert all(img and main.Path(img).is_file() for img in images)
    assert story["choices"], "the example should end with choices to continue from"
    assert out[-5] is not None  # its world is loaded too
