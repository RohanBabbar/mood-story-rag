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


EMILIA = {"name": "Emilia", "kind": "old retired detective", "hair": "silver bob hair",
          "clothing": "brown tweed coat", "feature": "reading glasses"}


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
    assert result["cast"]["Emilia"] == "old retired detective, silver bob hair, brown tweed coat, reading glasses"
    assert result["cast"]["Jasper"] == "grey cat, short fur, torn ear"
    assert result["choices"] == ["Emilia visits the docks", "Jasper vanishes", "A letter arrives"]


def test_finds_json_wrapped_in_chatter():
    result = analyze('Sure! Here it is:\n```json\n{"summary": "New summary."}\n```')
    assert result["summary"] == "New summary."


def test_malformed_reply_falls_back_safely():
    result = analyze("{not valid json", cast={"Mira": "young girl"})
    assert result == {"summary": "old summary", "cast": {"Mira": "young girl"}, "scene": "",
                      "scene_characters": [], "choices": [], "stand_ins": {}}


def test_established_looks_are_never_overwritten():
    result = analyze(json.dumps({"main_character": {**EMILIA, "hair": "CHANGED"}}),
                     cast={"Emilia": "retired detective, silver bob"})
    assert result["cast"] == {"Emilia": "retired detective, silver bob"}


def test_commas_inside_a_detail_are_removed():
    # A look is "kind, hair, clothing, feature"; a stray comma would split one detail into two.
    result = analyze(json.dumps({"main_character": {"name": "Mara", "kind": "young cartographer",
                                                    "hair": "brown, cropped hair"}}))
    assert result["cast"]["Mara"] == "young cartographer, brown cropped hair"


def test_ignores_objects_and_placeholder_details():
    result = analyze(json.dumps({"new_characters": [
        {"name": "the clock", "kind": "old clock"},
        {"name": "Night", "kind": "black cat", "clothing": "none", "feature": "n/a"},
    ]}))
    assert result["cast"] == {"Night": "black cat"}


# === Characters without a look: one follow-up, then a plain stand-in ===

LIRA = {"name": "Lira", "kind": "young traveller", "hair": "copper braided hair", "clothing": "green cloak",
        "feature": "scar on chin"}
CHAPTER = "Lira found the wreck. She knelt beside it. Bronn shouted, and he ran to her."


def analyze_with(replies, cast=None):
    """Run analyze_chapter against scripted model replies; returns (result, calls made)."""
    calls = []

    def fake_stream_chat(messages, **options):
        calls.append(options)
        reply = replies[len(calls) - 1]
        if isinstance(reply, Exception):
            raise reply
        yield reply

    with mock.patch.object(llm, "stream_chat", fake_stream_chat):
        return llm.analyze_chapter(CHAPTER, cast=cast), calls


def test_bare_name_protagonist_is_described_by_one_follow_up():
    result, calls = analyze_with([json.dumps({"main_character": "Lira", "scene_characters": ["Lira"]}),
                                  json.dumps({"characters": [LIRA]})])
    assert result["cast"]["Lira"] == "young traveller, copper braided hair, green cloak, scar on chin"
    assert result["stand_ins"] == {}
    # The follow-up is short and never tries other models.
    assert len(calls) == 2 and calls[1]["timeout"] == llm.DESCRIBE_TIMEOUT_SECONDS and calls[1]["fallback"] is False


def test_no_follow_up_when_every_character_has_a_look():
    _, calls = analyze_with([json.dumps({"main_character": LIRA, "scene_characters": ["Lira"]})])
    assert len(calls) == 1


@pytest.mark.parametrize("follow_up", [json.dumps({"characters": []}), TimeoutError("too slow")])
def test_a_failed_follow_up_leaves_a_plain_stand_in(follow_up):
    result, calls = analyze_with([json.dumps({"main_character": "Lira", "scene_characters": ["Lira", "Bronn"]}),
                                  follow_up])
    assert len(calls) == 2  # at most one follow-up
    assert result["stand_ins"] == {"Lira": "woman", "Bronn": "man"}
    assert "Lira" not in result["cast"]  # stand-ins are for one picture, never saved as a look


@pytest.mark.parametrize("text, kind", [
    ("Kai waved. He grinned.", "man"),
    ("Ember the fox sniffed the air. Its tail twitched.", "creature"),
    ("Sam arrived. Nobody spoke.", "person"),
    ("Ana saw Ben. She waved and he waved back.", "person"),  # a tie is too close to call
])
def test_stand_in_follows_the_pronouns_near_the_name(text, kind):
    assert llm.stand_in_look(text.split()[0], text) == kind


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

    failures, calls, api_key, base_url, extras = {}, [], None, None, []

    def __init__(self, provider=None, api_key=None, base_url=None, timeout=None):
        FakeClient.api_key, FakeClient.base_url = api_key, base_url

    def chat_completion(self, messages, model, extra_body=None, **kwargs):
        FakeClient.calls.append(model)
        FakeClient.extras.append(extra_body)
        failure = FakeClient.failures.get(model)
        if failure == "midway":
            def stream():
                yield chunk("Once ")
                raise http_error(503)
            return stream()
        if failure:
            raise http_error(failure)
        return iter([chunk("Once "), chunk("upon a time.")])


def stream_with(failures, primary=llm.TEXT_MODEL):
    FakeClient.failures, FakeClient.calls, FakeClient.extras = failures, [], []
    with mock.patch("huggingface_hub.InferenceClient", FakeClient):
        return "".join(llm._stream_api([], 10, 0.5, token="visitor-token", primary=primary))


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


@pytest.fixture
def groq(monkeypatch):
    monkeypatch.setattr(llm, "TEXT_PROVIDER", "groq")
    monkeypatch.setattr(llm, "GROQ_API_KEY", "gsk_test_not_real")
    monkeypatch.setattr(llm, "TEXT_FALLBACK_MODELS", ["openai/gpt-oss-20b", "qwen/qwen3.8-27b"])


def test_groq_uses_the_server_key_not_the_visitors_token(groq):
    assert stream_with({}, primary="openai/gpt-oss-120b") == "Once upon a time."
    assert FakeClient.base_url == llm.GROQ_BASE_URL
    assert FakeClient.api_key == "gsk_test_not_real"
    assert FakeClient.extras == [{"reasoning_effort": "low"}]


def test_groq_falls_back_on_a_per_model_rate_limit(groq):
    # Groq's limits are per model, so unlike on HF a 429 is worth retrying on the next model.
    assert stream_with({"openai/gpt-oss-120b": 429, "openai/gpt-oss-20b": 429},
                       primary="openai/gpt-oss-120b") == "Once upon a time."
    assert FakeClient.calls == ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
    assert FakeClient.extras[-1] == {"reasoning_effort": "none"}


def test_groq_errors_mention_groq_not_hugging_face(groq):
    with pytest.raises(llm.StoryError, match="GROQ_API_KEY"):
        stream_with({"openai/gpt-oss-120b": 401}, primary="openai/gpt-oss-120b")


def test_hf_requests_send_no_provider_extras():
    stream_with({})
    assert FakeClient.extras == [None]


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
