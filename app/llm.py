import json
import re
from contextlib import nullcontext
from functools import lru_cache
from threading import Thread

from app.config import (
    ADAPTER_REPO,
    ANALYSIS_MODEL,
    BACKEND,
    GROQ_API_KEY,
    GROQ_BASE_URL,
    TEXT_FALLBACK_MODELS,
    TEXT_MODEL,
    TEXT_PROVIDER,
)

# Rough word targets per chapter; tokens are budgeted with headroom so endings aren't cut off.
LENGTHS = {"Short": 150, "Medium": 280, "Long": 450}

# Only the most recent chapters are sent back to the model to keep the prompt small.
MAX_CONTEXT_CHAPTERS = 3


class StoryError(Exception):
    """A generation failure with a message that is safe to show to users."""


def build_messages(prompt, genre, mood, lore_context="", previous_chapters=(), length="Medium", cast=None, summary=""):
    words = LENGTHS.get(length, LENGTHS["Medium"])
    system = (
        "You are a gifted novelist writing an episodic, illustrated story.\n"
        f"Genre: {genre}. Mood: {mood or 'author’s choice'}.\n"
        "Rules for every chapter:\n"
        "- Something must happen: a discovery, a decision, a conflict or a change that moves the plot "
        "forward. Description supports the action; it never replaces it.\n"
        f"- The {genre} genre must show in the events themselves, not just the scenery.\n"
        "- Let the mood shape word choice, pacing and what characters notice.\n"
        "- Name the main character in the first paragraph, and introduce new characters by name.\n"
        "- Include at least one line of dialogue.\n"
        "- Never reuse distinctive phrases or images from earlier chapters.\n"
        f"- About {words} words, in short paragraphs.\n"
        "Output only the story text as plain prose: no Markdown, no title, no chapter heading, no notes or "
        "commentary. "
        "End on a moment that invites the story to continue.\n"
        "Keep it suitable for a general audience: no explicit sexual content and no graphic gore. "
        "Even in horror, imply violence rather than describe it in detail."
    )
    if cast:
        system += "\n\nCharacters so far (keep them consistent):\n" + "\n".join(
            f"- {name}: {look}" for name, look in cast.items()
        )
    if lore_context:
        system += (
            "\n\nWorld lore: treat the following as canon. Use names, places and rules from it, "
            "and never contradict it.\n<lore>\n" + lore_context + "\n</lore>"
        )

    if previous_chapters:
        story_so_far = "\n\n".join(previous_chapters[-MAX_CONTEXT_CHAPTERS:])
        # Older chapters fall out of the window, so the running summary stands in for them.
        earlier = (f"Summary of earlier events: {summary}\n\n"
                   if summary and len(previous_chapters) > MAX_CONTEXT_CHAPTERS else "")
        user = earlier + (
            f"The most recent chapters:\n<story>\n{story_so_far}\n</story>\n\n"
            "Write the next chapter, picking up exactly where it left off without recapping. "
            "Introduce a new development that changes the situation."
            + (f"\nWhat should happen next: {prompt}" if prompt else "")
        )
    else:
        user = f"Begin a new story from this premise: {prompt}"

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def stream_story(messages, length="Medium", token=None, use_adapter=True):
    """Yield the story text incrementally as it is generated.

    `use_adapter` picks the fine-tuned LoRA (local backend only) or the plain base model.
    """
    # Generous budget (~1.3 tokens per word, plus slack): models overshoot the target length, and
    # anything that still runs over is trimmed to the last full sentence by trim_to_last_sentence.
    max_tokens = int(LENGTHS.get(length, LENGTHS["Medium"]) * 2) + 120
    # A mild repetition penalty stops small local models looping on the same phrases.
    yield from stream_chat(messages, max_tokens, token=token, repetition_penalty=1.1, use_adapter=use_adapter)


# A sentence ends with . ! ? or …, optionally followed by closing quotes or brackets.
_SENTENCE_END = re.compile(r"""[.!?…]["'”’)\]]*""")


def trim_to_last_sentence(text):
    """Drop the half-sentence left behind when generation hits its token limit."""
    text = text.strip()
    ends = list(_SENTENCE_END.finditer(text))
    if not ends or ends[-1].end() == len(text):
        return text
    cut = ends[-1].end()
    # If trimming would throw away a big chunk, keep it and mark the break instead.
    if cut < len(text) * 0.6:
        return text + "…"
    return text[:cut]


def stream_chat(messages, max_tokens, temperature=0.85, token=None, repetition_penalty=1.0, use_adapter=False,
                model=None):
    """`token` is the signed-in visitor's HF token; None falls back to the server's HF_TOKEN. It is
    ignored on Groq, which always uses the server's GROQ_API_KEY. `model` overrides TEXT_MODEL for
    hosted backends.

    `repetition_penalty` and `use_adapter` only apply to local models. Keep the penalty at 1.0 for
    JSON output, whose brackets and quotes legitimately repeat; the adapter is off by default
    because it was trained to write stories, not JSON.
    """
    if BACKEND == "local":
        yield from _stream_local(messages, max_tokens, temperature, repetition_penalty, use_adapter)
    else:
        yield from _stream_api(messages, max_tokens, temperature, token, model or TEXT_MODEL)


ANALYSIS_PROMPT = """Story summary before this chapter: {summary}

Known characters (name: appearance):
{cast}

Newest chapter:
<chapter>
{chapter}
</chapter>

Reply with only a JSON object with these keys:
"summary": 2-4 sentences summarising the whole story so far, including the newest chapter.
"main_character": the story's protagonist, always filled in, even if already known. \
"new_characters": one entry for every other named character in the newest chapter who is not already a known \
character. Every character entry is \
{{"name": ..., "kind": ..., "age": ..., "hair": ..., "clothing": ..., "feature": ...}} where \
"kind" is what they are in two or three words, "age" is an approximate age, "hair" is hair colour and style (fur for \
animals), "clothing" is their main garment and its colour, and "feature" is one distinctive physical detail. \
Use plain physical words, not metaphors from the story. When the text doesn't say, invent details that suit \
this particular story and its setting.
"scene": one sentence describing the most striking visual moment of the newest chapter, for an illustrator. Refer to characters by name. Describe only what can be seen; don't name places and don't mention signs or writing.
"scene_characters": names of the characters visible in that scene.
"choices": three short, clearly different directions for the next chapter, each under 12 words, starting with a character's name and an action."""


def analyze_chapter(chapter, summary="", cast=None, token=None):
    """Ask the model for a story summary, cast list, illustration scene and next-step choices.

    Returns a dict with keys summary, cast, scene, scene_characters, choices. Missing or malformed
    fields fall back to safe defaults so a bad reply never breaks the story.
    """
    cast = dict(cast or {})
    messages = [
        {"role": "system", "content": "You are a meticulous story editor. You reply with valid JSON only."},
        {"role": "user", "content": ANALYSIS_PROMPT.format(
            summary=summary or "(this is the first chapter)",
            cast="\n".join(f"- {n}: {a}" for n, a in cast.items()) or "(none yet)",
            chapter=chapter,
        )},
    ]
    reply = "".join(stream_chat(messages, max_tokens=600, temperature=0.4, token=token, model=ANALYSIS_MODEL))
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}

    # The protagonist gets its own key because small models skip it when it's just one list item.
    people = [data.get("main_character")] + list(data.get("new_characters") or [])
    for person in people:
        name = str(person.get("name") or "").strip() if isinstance(person, dict) else ""
        # Proper names only; the model sometimes lists objects like "the clock".
        if not name or not name[0].isupper():
            continue
        details = (str(person.get(k) or "").strip() for k in ("kind", "age", "hair", "clothing", "feature"))
        look = ", ".join(d for d in details if d and d.lower() not in ("none", "n/a", "unknown"))
        if look:
            # Never overwrite an established look; that's the whole point of the cast list.
            cast.setdefault(name, look)

    choices = [str(c).strip() for c in data.get("choices") or [] if str(c).strip()][:3]
    return {
        "summary": str(data.get("summary") or summary).strip(),
        "cast": cast,
        "scene": str(data.get("scene") or "").strip(),
        "scene_characters": [str(n) for n in data.get("scene_characters") or []],
        "choices": choices,
    }


# Statuses that mean "this model is unavailable", where a backup model may still work. Account problems
# (401/402/403) would fail the same way on every model, so those aren't retried. Groq's rate limits
# are per model, so there a 429 is worth retrying on the next model too.
FALLBACK_STATUSES = {400, 404, 410, 422, 500, 502, 503, 504}
GROQ_FALLBACK_STATUSES = FALLBACK_STATUSES | {429}


def status_of(error):
    return getattr(getattr(error, "response", None), "status_code", None)


def models_to_try(primary, fallbacks):
    return [primary] + [m for m in fallbacks if m != primary]


def request_options(model):
    """Provider-specific extras. gpt-oss and Qwen 3 think before answering; on Groq that thinking
    arrives separately from the text and uses up the token budget, so keep it short or off.
    """
    if TEXT_PROVIDER != "groq":
        return None
    if model.startswith("openai/gpt-oss"):
        return {"reasoning_effort": "low"}
    if model.startswith("qwen/qwen3"):
        return {"reasoning_effort": "none"}
    return None


def _stream_api(messages, max_tokens, temperature, token=None, primary=TEXT_MODEL):
    from huggingface_hub import InferenceClient
    from huggingface_hub.errors import HfHubHTTPError

    if TEXT_PROVIDER == "groq":
        # Groq speaks the OpenAI chat API, which InferenceClient can call directly.
        client = InferenceClient(base_url=GROQ_BASE_URL, api_key=GROQ_API_KEY)
        retry_statuses = GROQ_FALLBACK_STATUSES
    else:
        client = InferenceClient(provider="auto", api_key=token)
        retry_statuses = FALLBACK_STATUSES
    error = None
    for model in models_to_try(primary, TEXT_FALLBACK_MODELS):
        started = False
        try:
            for chunk in client.chat_completion(
                messages, model=model, stream=True, max_tokens=max_tokens, temperature=temperature, top_p=0.95,
                extra_body=request_options(model),
            ):
                if chunk.choices and chunk.choices[0].delta.content:
                    started = True
                    yield chunk.choices[0].delta.content
            return
        except HfHubHTTPError as e:
            error = e
            # Switching models halfway through a chapter would produce a jarring seam, so only
            # fall back before any text has been shown.
            if started or status_of(e) not in retry_statuses:
                break
            print(f"[text] {model} failed with {status_of(e)}; trying the next model")
    if TEXT_PROVIDER == "groq":
        raise StoryError(explain_groq_error(error)) from error
    raise StoryError(explain_hf_error(error)) from error


def explain_groq_error(e):
    status = status_of(e)
    if status == 401:
        return "Groq rejected the API key. Set a valid GROQ_API_KEY."
    if status == 429:
        return "The free story models are busy right now. Wait a minute and try again."
    return f"Model request failed ({status or 'network error'}). Please try again."


def explain_hf_error(e):
    status = status_of(e)
    if status == 401:
        return "Hugging Face rejected the token. Set a valid HF_TOKEN."
    if status == 403:
        return ("The HF token lacks permission to call Inference Providers. Create a fine-grained token "
                "with “Make calls to Inference Providers” enabled.")
    if status == 402:
        return "Hugging Face inference credits are used up for this month."
    if status == 429:
        return "Rate limited by Hugging Face. Wait a moment and try again."
    return f"Model request failed ({status or 'network error'}). Please try again."


@lru_cache(maxsize=1)
def _load_local_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        TEXT_MODEL, dtype=torch.float16 if device != "cpu" else torch.float32
    ).to(device)
    if ADAPTER_REPO:
        from peft import PeftModel

        # The adapter was trained on a 4-bit (bitsandbytes) copy of this base. LoRA weights are
        # independent of the base's storage precision, so they load onto fp16 without bitsandbytes,
        # which only runs on NVIDIA GPUs.
        model = PeftModel.from_pretrained(model, ADAPTER_REPO)
    model.eval()
    return tokenizer, model


def adapter_switch(model, use_adapter):
    """Context in which generation uses the fine-tuned adapter or the plain base model.

    One model in memory serves both: PEFT can bypass the LoRA weights per call.
    """
    if hasattr(model, "disable_adapter") and not use_adapter:
        return model.disable_adapter()
    return nullcontext()


def _stream_local(messages, max_tokens, temperature, repetition_penalty=1.0, use_adapter=False):
    from transformers import TextIteratorStreamer

    try:
        tokenizer, model = _load_local_model()
    except Exception as e:
        raise StoryError(f"Could not load local model {TEXT_MODEL}: {e}") from e

    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    # skip_prompt ensures only newly generated text is returned, not the instructions.
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    errors = []

    def generate():
        try:
            with adapter_switch(model, use_adapter):
                model.generate(**inputs, streamer=streamer, max_new_tokens=max_tokens, do_sample=True,
                               temperature=temperature, top_p=0.95, repetition_penalty=repetition_penalty,
                               pad_token_id=tokenizer.eos_token_id)
        except Exception as e:
            errors.append(e)
            streamer.end()  # otherwise the reader below would wait for text forever

    Thread(target=generate, daemon=True).start()
    yield from streamer
    if errors:
        raise StoryError(f"Local generation failed: {errors[0]}") from errors[0]
