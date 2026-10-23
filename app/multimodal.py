import re
import threading
import uuid
from functools import lru_cache

from app.config import IMAGE_FALLBACK_MODELS, IMAGE_MODEL, IMAGE_PROVIDER, OUTPUT_DIR

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


def build_image_prompt(scene, genre, mood, style=DEFAULT_ART_STYLE, characters=()):
    """Build an illustration prompt from a scene, the look of the characters in it, and the art style.

    Image models don't know who "Lila" is: a name next to a separate description gets drawn as two
    people, or printed as text on the picture. So each name is replaced by the character's fixed
    description right where it appears, which also keeps them recognisable across chapters.
    """
    for name, look in characters:
        kind, _, details = (part.strip() for part in look.partition(","))
        described = _with_article(kind) + (f" ({details})" if details else "")
        pattern = re.compile(rf"\b{re.escape(name)}('s)?\b")
        if pattern.search(scene):
            scene = pattern.sub(lambda m, d=described: d + (m.group(1) or ""), scene, count=1)
            scene = pattern.sub(lambda m, k=kind: f"the {k}" + (m.group(1) or ""), scene)
        else:
            scene += f" Also in the scene: {described}."
    style_prompt = ART_STYLES.get(style) or ART_STYLES[DEFAULT_ART_STYLE]
    # Stable Diffusion models only read the first 77 tokens, so the core of the style ("watercolor
    # illustration") goes first, then the scene, and the finer style details last, where they are
    # the first thing dropped.
    style_head, _, style_rest = style_prompt.partition(", ")
    tail = ", ".join(t for t in (style_rest, f"{genre.lower()} story", f"{mood.lower()} atmosphere") if t)
    return f"{style_head}. {scene[:600]} {tail}"


def _with_article(phrase):
    if re.match(r"(a|an|the)\s", phrase, re.IGNORECASE):
        return phrase
    return f"an {phrase}" if phrase[:1].lower() in "aeiou" else f"a {phrase}"


def first_sentences(text, n=2):
    """Fallback scene when no better description is available."""
    return " ".join(re.split(r"(?<=[.!?])\s+", text.strip())[:n])


def generate_image(scene, genre, mood, style=DEFAULT_ART_STYLE, characters=(), token=None):
    """Illustrate a scene. Returns the image path, or raises on failure."""
    prompt = build_image_prompt(scene, genre, mood, style, characters)
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


@lru_cache(maxsize=1)
def load_image_pipeline():
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
