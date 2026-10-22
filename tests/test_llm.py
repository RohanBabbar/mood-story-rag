import json
import types
from unittest import mock

import pytest
from huggingface_hub.errors import HfHubHTTPError

from app import llm


def http_error(status):
    return HfHubHTTPError(f"HTTP {status}", response=mock.Mock(status_code=status, headers={}))


# === Prompt building ===

def test_first_chapter_starts_from_the_premise():
    system, user = llm.build_messages("A door in the sea", "Fantasy", "Eerie")
    assert "Genre: Fantasy. Mood: Eerie." in system["content"]
    assert "Something must happen" in system["content"]
    assert user["content"] == "Begin a new story from this premise: A door in the sea"


def test_continuation_includes_recent_chapters_and_direction():
    _, user = llm.build_messages("The wyrm wakes", "Fantasy", "Eerie", previous_chapters=["one", "two"])
    assert "one\n\ntwo" in user["content"]
    assert "What should happen next: The wyrm wakes" in user["content"]


def test_summary_replaces_chapters_that_fall_out_of_the_window():
    chapters = [f"chapter {i}" for i in range(llm.MAX_CONTEXT_CHAPTERS + 2)]
    _, user = llm.build_messages("", "Fantasy", "Eerie", previous_chapters=chapters, summary="Earlier events.")
    assert user["content"].startswith("Summary of earlier events: Earlier events.")
    assert "chapter 0" not in user["content"]


def test_cast_and_lore_are_given_to_the_writer():
    system, _ = llm.build_messages("x", "Fantasy", "Eerie", lore_context="Magic runs on songs.",
                                   cast={"Mira": "young girl, red hair"})
    assert "- Mira: young girl, red hair" in system["content"]
    assert "<lore>\nMagic runs on songs.\n</lore>" in system["content"]


# === Trimming cut-off chapters ===

LONG = "The banana quivered and Gorvoth held his breath. " * 10


@pytest.mark.parametrize("text, expected_end", [
    (LONG + "A portal opened, drawing everyone in the", "held his breath."),
    ('She smiled. "Hold onto your chairs."', '"Hold onto your chairs."'),
    ("The banana began to quiver...", "quiver..."),
    ("a fragment with no ending", "a fragment with no ending"),
])
def test_trim_to_last_sentence(text, expected_end):
    assert llm.trim_to_last_sentence(text).endswith(expected_end)


def test_trim_marks_the_break_instead_of_dropping_most_of_the_text():
    text = "Short. " + "and the clause kept going without a stop " * 5
    assert llm.trim_to_last_sentence(text) == text.strip() + "…"


# === Parsing the analysis JSON ===

def analyze(reply, cast=None):
    with mock.patch.object(llm, "stream_chat", lambda *a, **k: iter([reply])):
        return llm.analyze_chapter("chapter text", summary="old summary", cast=cast)


EMILIA = {"name": "Emilia", "kind": "retired detective", "age": "about sixty", "hair": "silver bob",
          "clothing": "tweed coat", "feature": "reading glasses"}


def test_parses_a_full_reply():
    result = analyze(json.dumps({
        "summary": "Emilia finds a clue.",
        "main_character": EMILIA,
        "new_characters": [{"name": "Jasper", "kind": "grey cat", "hair": "short fur", "feature": "torn ear"}],
        "scene": "Emilia holds a muddy receipt.",
        "scene_characters": ["Emilia"],
        "choices": ["Emilia visits the docks", "Jasper vanishes", "A letter arrives", "a fourth one"],
    }))
    assert result["summary"] == "Emilia finds a clue."
    assert result["cast"]["Emilia"] == "retired detective, about sixty, silver bob, tweed coat, reading glasses"
    assert result["cast"]["Jasper"] == "grey cat, short fur, torn ear"
    assert result["choices"] == ["Emilia visits the docks", "Jasper vanishes", "A letter arrives"]


def test_finds_json_wrapped_in_chatter():
    result = analyze('Sure! Here it is:\n```json\n{"summary": "New summary."}\n```')
    assert result["summary"] == "New summary."


def test_malformed_reply_falls_back_safely():
    result = analyze("{not valid json", cast={"Mira": "young girl"})
    assert result == {"summary": "old summary", "cast": {"Mira": "young girl"}, "scene": "",
                      "scene_characters": [], "choices": []}


def test_established_looks_are_never_overwritten():
    result = analyze(json.dumps({"main_character": {**EMILIA, "hair": "CHANGED"}}),
                     cast={"Emilia": "retired detective, silver bob"})
    assert result["cast"] == {"Emilia": "retired detective, silver bob"}


def test_ignores_objects_and_placeholder_details():
    result = analyze(json.dumps({"new_characters": [
        {"name": "the clock", "kind": "old clock"},
        {"name": "Night", "kind": "black cat", "clothing": "none", "feature": "n/a"},
    ]}))
    assert result["cast"] == {"Night": "black cat"}


# === Hosted models: errors and fallbacks ===

@pytest.mark.parametrize("status, phrase", [
    (401, "rejected the token"), (402, "credits are used up"), (403, "lacks permission"), (429, "Rate limited"),
])
def test_http_errors_become_readable_messages(status, phrase):
    assert phrase in llm.explain_hf_error(http_error(status))


def chunk(text):
    return types.SimpleNamespace(choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content=text))])


class FakeClient:
    """Stands in for InferenceClient. `failures` maps model name -> HTTP status or "midway"."""

    failures, calls, api_key = {}, [], None

    def __init__(self, provider=None, api_key=None):
        FakeClient.api_key = api_key

    def chat_completion(self, messages, model, **kwargs):
        FakeClient.calls.append(model)
        failure = FakeClient.failures.get(model)
        if failure == "midway":
            def stream():
                yield chunk("Once ")
                raise http_error(503)
            return stream()
        if failure:
            raise http_error(failure)
        return iter([chunk("Once "), chunk("upon a time.")])


def stream_with(failures):
    FakeClient.failures, FakeClient.calls = failures, []
    with mock.patch("huggingface_hub.InferenceClient", FakeClient):
        return "".join(llm._stream_api([], 10, 0.5, token="visitor-token"))


PRIMARY, BACKUP = llm.TEXT_MODEL, llm.TEXT_FALLBACK_MODELS[0]


def test_uses_the_primary_model_and_the_callers_token():
    assert stream_with({}) == "Once upon a time."
    assert FakeClient.calls == [PRIMARY]
    assert FakeClient.api_key == "visitor-token"


def test_falls_back_when_the_model_is_unavailable():
    assert stream_with({PRIMARY: 404}) == "Once upon a time."
    assert FakeClient.calls == [PRIMARY, BACKUP]


def test_does_not_fall_back_on_account_problems():
    # Out of credits fails the same way on every model, so retrying would only waste time.
    with pytest.raises(llm.StoryError, match="credits"):
        stream_with({PRIMARY: 402})
    assert FakeClient.calls == [PRIMARY]


def test_does_not_switch_models_halfway_through_a_chapter():
    with pytest.raises(llm.StoryError):
        stream_with({PRIMARY: "midway"})
    assert FakeClient.calls == [PRIMARY]


# === Local model: adapter switch and background-thread failures ===

class FakePeftModel:
    def disable_adapter(self):
        return "adapter disabled"


def test_adapter_is_bypassed_only_when_asked():
    assert llm.adapter_switch(FakePeftModel(), use_adapter=False) == "adapter disabled"
    assert llm.adapter_switch(FakePeftModel(), use_adapter=True) != "adapter disabled"
    # A plain model without an adapter just runs as itself.
    assert llm.adapter_switch(object(), use_adapter=False) is not None


class FakeTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, *args, **kwargs):
        return types.SimpleNamespace(to=lambda device: {})


def run_local(generate):
    model = types.SimpleNamespace(device="cpu", generate=generate)
    with mock.patch.object(llm, "_load_local_model", lambda: (FakeTokenizer(), model)):
        return "".join(llm._stream_local([], 10, 0.5))


def test_local_generation_streams_text():
    def generate(streamer, **kwargs):
        streamer.on_finalized_text("Once upon a time.", stream_end=True)
    assert run_local(generate) == "Once upon a time."


def test_local_generation_crash_is_reported_instead_of_hanging():
    # Regression: an exception in the generation thread used to leave the reader waiting forever.
    def generate(**kwargs):
        raise RuntimeError("MPS backend out of memory")
    with pytest.raises(llm.StoryError, match="out of memory"):
        run_local(generate)
