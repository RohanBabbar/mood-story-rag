import logging
import re
import threading
import uuid
from functools import lru_cache

from app.config import IMAGE_FALLBACK_MODELS, IMAGE_MODEL, IMAGE_PROVIDER, OUTPUT_DIR

logger = logging.getLogger(__name__)

NO_ILLUSTRATIONS = "No illustrations"

# Each style is appended to the scene description; keep them as visual descriptors the image model understands.
ART_STYLES = {
    "Storybook painting": "storybook illustration, detailed digital painting, cinematic lighting",
    "Hand-drawn anime": "hand-drawn Japanese anime film still, soft watercolor backgrounds, lush nature, "
                        "warm gentle light, whimsical",
    "Watercolor": "loose watercolor illustration, soft color washes, visible paper texture, delicate ink outlines",
    "Comic book": "comic book panel, bold ink linework, halftone shading, vivid flat colors, dynamic composition",
    "3D animated film": "3D animated movie still, stylized expressive characters, soft global illumination, vibrant colors",
    "Cinematic photo": "cinematic film still, photorealistic, 35mm lens, shallow depth of field, dramatic lighting",
    "Oil painting": "classical oil painting, visible brushstrokes, rich deep colors, chiaroscuro lighting",
    "Pixel art": "detailed 16-bit pixel art scene, limited color palette, retro video game style",
    "Ink & charcoal": "black and white ink and charcoal illustration, high contrast, cross-hatching, moody",
    NO_ILLUSTRATIONS: None,
}
DEFAULT_ART_STYLE = "Storybook painting"


# Words in a name that say how a character is addressed, not who they are: "Keeper Vesh" is often
# just "Vesh" in the text, and "the Keeper" could be anyone.
TITLES = frozenset({
    "the", "of", "mr", "mrs", "ms", "miss", "mx", "dr", "doctor", "sir", "dame", "lord", "lady", "king",
    "queen", "prince", "princess", "duke", "duchess", "emperor", "empress", "captain", "commander", "general",
    "lieutenant", "sergeant", "admiral", "keeper", "master", "mistress", "father", "mother", "brother",
    "sister", "uncle", "aunt", "grandma", "grandpa", "saint", "st", "professor", "detective", "inspector",
    "officer", "agent", "elder",
})


def _name_words(name):
    return [w for w in re.findall(r"\w[\w'’-]*", name) if w[:1].isupper() and w.lower().rstrip(".") not in TITLES]


def name_aliases(names):
    """Map every way the text may refer to a character to that character's full name.

    A full name always belongs to its character. Shorter forms (the name without titles, and each
    remaining word) are added only when they point to exactly one character: if "Mara Lorn" and
    "Ellis Lorn" are both in the story, "Lorn" alone is ambiguous and matches neither.
    """
    aliases = {name: name for name in names}
    claims = {}
    for name in names:
        words = _name_words(name)
        for alias in {" ".join(words), *words} - {""}:
            claims.setdefault(alias, set()).add(name)
    for alias, owners in claims.items():
        if alias not in aliases and len(owners) == 1:
            aliases[alias] = next(iter(owners))
    return aliases


def _mention_pattern(aliases):
    # Longest alias first, because a regex alternation takes the first one that matches: "Mara Lorn"
    # must win over "Mara". Whole words only, so "Ann" doesn't match inside "Annie".
    options = "|".join(re.escape(a) for a in sorted(aliases, key=len, reverse=True))
    return re.compile(rf"(?<![\w'’-])({options})(['’]s)?(?![\w'’-])")


def match_cast_name(name, cast_names):
    """The cast member a name refers to ("Mara" -> "Mara Lorn"), or None if unknown or ambiguous."""
    aliases = name_aliases(cast_names)
    if name in aliases:
        return aliases[name]
    words = _name_words(name)
    for candidate in sorted({" ".join(words), *words} - {""}, key=len, reverse=True):
        if candidate in aliases:
            return aliases[candidate]
    return None


def scene_cast(scene, listed_names, cast):
    """The (name, look) of every cast member in the scene: those it mentions, in order of appearance,
    then any others the analysis listed as visible."""
    if not cast:
        return []
    mentioned = [name_aliases(cast)[m.group(1)] for m in _mention_pattern(name_aliases(cast)).finditer(scene)]
    listed = [match_cast_name(n, list(cast)) for n in listed_names]
    return [(name, cast[name]) for name in dict.fromkeys(mentioned + listed) if name]


# Stable Diffusion's text encoder reads 77 tokens: 75 of prompt plus start and end markers.
CLIP_MAX_TOKENS = 75

# When a prompt is too long, these are applied one after another until it fits, so the least
# important parts go first: genre and mood, then the finer style words, then the third character's
# details, then each character's distinctive feature, then characters the scene doesn't mention, and
# last every remaining detail. The core style, the scene's action and who is in it always stay.
_SHRINK_STEPS = [
    {"genre_mood": False}, {"fine_style": False}, {"detailed": 2}, {"details": 2}, {"extras": False},
    {"detailed": 0},
]


def build_image_prompt(scene, genre, mood, style=DEFAULT_ART_STYLE, characters=(), count_tokens=None,
                       max_tokens=CLIP_MAX_TOKENS):
    """Build an illustration prompt from a scene, the look of the characters in it, and the art style.

    Image models don't know who "Lila" is: a name next to a separate description gets drawn as two
    people, or printed as text on the picture. So each name is replaced by the character's fixed
    description right where it appears, which also keeps them recognisable across chapters.

    `count_tokens` is the image model's token counter. When given, the prompt is shrunk until it fits
    in `max_tokens`, because anything past the limit is silently ignored by the model.
    """
    style_prompt = ART_STYLES.get(style) or ART_STYLES[DEFAULT_ART_STYLE]
    style_head, _, style_details = style_prompt.partition(", ")
    options = {"genre_mood": True, "fine_style": True, "detailed": 3, "details": None, "extras": True}
    for step in [{}, *_SHRINK_STEPS]:
        options.update(step)
        prompt = _render_prompt(scene, genre, mood, style_head, style_details, characters, **options)
        if count_tokens is None or count_tokens(prompt) <= max_tokens:
            break
    return prompt


def _render_prompt(scene, genre, mood, style_head, style_details, characters, genre_mood, fine_style, detailed,
                   details, extras):
    """One version of the prompt. `detailed` characters get their details (all, or the first
    `details` of them); the rest are named by kind only."""
    kinds, described = {}, {}
    for i, (name, look) in enumerate(characters):
        kind, *rest = [part.strip() for part in look.split(",") if part.strip()] or [name]
        rest = rest[:details] if details is not None else rest
        kinds[name] = kind
        described[name] = _with_article(kind) + (f" ({', '.join(rest)})" if rest and i < detailed else "")

    aliases = name_aliases(list(described))
    seen = set()

    def describe(match):
        name = aliases[match.group(1)]
        text = f"the {kinds[name]}" if name in seen else described[name]
        seen.add(name)
        return text + (match.group(2) or "")

    if aliases:
        scene = _mention_pattern(aliases).sub(describe, scene)
    others = [described[n] for n in described if n not in seen]
    if others and extras:
        scene += f" Also in the scene: {', '.join(others)}."
    tail = [style_details if fine_style else "",
            *((f"{genre.lower()} story", f"{mood.lower()} atmosphere") if genre_mood else ())]
    tail = ", ".join(t for t in tail if t)
    return f"{style_head}. {scene[:600]}" + (f" {tail}" if tail else "")


def _with_article(phrase):
    if re.match(r"(a|an|the)\s", phrase, re.IGNORECASE):
        return phrase
    return f"an {phrase}" if phrase[:1].lower() in "aeiou" else f"a {phrase}"


def first_sentences(text, n=2):
    """Fallback scene when no better description is available."""
    return " ".join(re.split(r"(?<=[.!?])\s+", text.strip())[:n])


def generate_image(scene, genre, mood, style=DEFAULT_ART_STYLE, characters=(), token=None):
    """Illustrate a scene. Returns the image path, or raises on failure."""
    # Hosted FLUX reads long prompts; the Stable Diffusion models drawn here stop at 77 tokens.
    count_tokens = None if IMAGE_PROVIDER == "hf" else count_clip_tokens
    prompt = build_image_prompt(scene, genre, mood, style, characters, count_tokens=count_tokens)
    # One line per picture, so a mismatched illustration can be traced to its scene or its prompt.
    tokens = count_tokens(prompt) if count_tokens else "n/a"
    logger.info("Illustrating scene=%r characters=%r prompt=%r tokens=%s",
                scene, [name for name, _ in characters], prompt, tokens)
    image = _generate_api(prompt, token) if IMAGE_PROVIDER == "hf" else _generate_in_process(prompt)
    path = OUTPUT_DIR / f"illustration_{uuid.uuid4().hex[:10]}.jpg"
    image.convert("RGB").save(path, quality=90)
    return str(path)


def _generate_api(prompt, token=None):
    from huggingface_hub import InferenceClient
    from huggingface_hub.errors import HfHubHTTPError

    from app.llm import FALLBACK_STATUSES, models_to_try, status_of

    client = InferenceClient(provider="auto", api_key=token)
    error = None
    for model in models_to_try(IMAGE_MODEL, IMAGE_FALLBACK_MODELS):
        try:
            # 3:2 matches the frame the story page shows, so nothing gets cropped off.
            return client.text_to_image(prompt, model=model, width=1152, height=768)
        except HfHubHTTPError as e:
            error = e
            if status_of(e) not in FALLBACK_STATUSES:
                break
            print(f"[image] {model} failed with {status_of(e)}; trying the next model")
    raise error


# SD-Turbo is distilled to draw in a single step and without classifier-free guidance; regular
# Stable Diffusion needs ~20 steps (a good speed/quality trade-off versus the default 50).
PIPELINE_SETTINGS = {
    "sd-turbo": {"num_inference_steps": 1, "guidance_scale": 0.0},
    "local": {"num_inference_steps": 20},
}

# One picture at a time: parallel runs on a small CPU only slow each other down and multiply memory.
_pipeline_lock = threading.Lock()


_load_lock = threading.Lock()


def load_image_pipeline():
    """The image model, loaded once.

    The startup preload and the first visitor's request can ask for it at the same moment; without
    the lock both would load it, doubling memory right after a cold start.
    """
    with _load_lock:
        return _load_image_pipeline()


@lru_cache(maxsize=1)
def _load_image_pipeline():
    import torch
    from diffusers import AutoPipelineForText2Image

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    pipe = AutoPipelineForText2Image.from_pretrained(
        IMAGE_MODEL, dtype=torch.float16 if device != "cpu" else torch.float32
    ).to(device)
    pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def count_clip_tokens(text):
    """Prompt length as the image model's own tokenizer sees it, without the start and end markers."""
    return len(load_image_pipeline().tokenizer(text, verbose=False).input_ids) - 2


def _generate_in_process(prompt):
    settings = PIPELINE_SETTINGS.get(IMAGE_PROVIDER, PIPELINE_SETTINGS["local"])
    with _pipeline_lock:
        return load_image_pipeline()(prompt, width=768, height=512, **settings).images[0]


def generate_narration(text):
    """Synthesize speech for the given text with gTTS and return the mp3 path."""
    from gtts import gTTS

    path = OUTPUT_DIR / f"narration_{uuid.uuid4().hex[:10]}.mp3"
    gTTS(text=text).save(str(path))
    return str(path)
