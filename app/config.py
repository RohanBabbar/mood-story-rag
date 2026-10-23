import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

# huggingface_hub reads HF_TOKEN; keep supporting the older HUGGINGFACE_TOKEN name.
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

# "api"   -> hosted models (no GPU needed, used for deployment)
# "local" -> run LLaMA + Stable Diffusion on this machine (needs requirements-local.txt)
BACKEND = os.getenv("MODEL_BACKEND", "api").lower()

# Who serves text for the api backend: Groq's free tier when a GROQ_API_KEY is set (visitors pay
# nothing, so no sign-in is needed), otherwise Hugging Face Inference Providers.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TEXT_PROVIDER = "local" if BACKEND == "local" else os.getenv("TEXT_PROVIDER", "groq" if GROQ_API_KEY else "hf").lower()
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

_TEXT_DEFAULTS = {
    "groq": "openai/gpt-oss-120b",
    "hf": "meta-llama/Llama-3.1-8B-Instruct",
    "local": "unsloth/Llama-3.2-3B-Instruct",
}
TEXT_MODEL = os.getenv("TEXT_MODEL", _TEXT_DEFAULTS.get(TEXT_PROVIDER, _TEXT_DEFAULTS["hf"]))
# The JSON analysis (summary, cast, scene, choices) is an easier job, so on Groq a smaller, faster
# model does it; this also keeps it off the story model's per-model rate limit.
ANALYSIS_MODEL = os.getenv("ANALYSIS_MODEL", "openai/gpt-oss-20b" if TEXT_PROVIDER == "groq" else TEXT_MODEL)

# Who draws the illustrations:
# "hf"       -> Hugging Face Inference Providers (FLUX; needs HF credits)
# "sd-turbo" -> SD-Turbo in this process: one denoising step, fast enough for a free CPU Space
#               (~30 s per picture), so it is the default alongside Groq, which has no image models
# "local"    -> Stable Diffusion 1.5 on this machine's GPU
IMAGE_PROVIDER = "local" if BACKEND == "local" else os.getenv(
    "IMAGE_PROVIDER", "sd-turbo" if TEXT_PROVIDER == "groq" else "hf").lower()
_IMAGE_DEFAULTS = {
    "hf": "black-forest-labs/FLUX.1-schnell",
    "sd-turbo": "stabilityai/sd-turbo",
    "local": "stable-diffusion-v1-5/stable-diffusion-v1-5",
}
IMAGE_MODEL = os.getenv("IMAGE_MODEL", _IMAGE_DEFAULTS.get(IMAGE_PROVIDER, _IMAGE_DEFAULTS["hf"]))
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Optional LoRA adapter applied on top of the local base model; when set, the UI offers a
# "Story model" toggle between the adapter and the plain base. Off by default: the current adapter
# (RohanBabbar/lora-finetuned-mood-story) changes the base weights by ~0.007%, so its output is
# identical to the base model's. unsloth/Llama-3.2-3B-Instruct is the ungated copy of its base.
ADAPTER_REPO = os.getenv("ADAPTER_REPO", "") if BACKEND == "local" else ""

OUTPUT_DIR = Path(tempfile.gettempdir()) / "mood-story"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Backups tried in order when the main model is unavailable (removed, overloaded or unsupported).
# Set to an empty string to disable.
_TEXT_FALLBACKS = "openai/gpt-oss-20b,qwen/qwen3.8-27b" if TEXT_PROVIDER == "groq" else "meta-llama/Llama-3.3-70B-Instruct"
TEXT_FALLBACK_MODELS = [m for m in os.getenv("TEXT_FALLBACK_MODELS", _TEXT_FALLBACKS).split(",") if m]
IMAGE_FALLBACK_MODELS = [m for m in os.getenv("IMAGE_FALLBACK_MODELS", "stabilityai/stable-diffusion-xl-base-1.0").split(",") if m]

# Hugging Face Spaces sets SPACE_ID. When text runs on HF Inference Providers, visitors to a Space
# sign in with Hugging Face by default so inference is billed to their own account, not the owner's.
ON_SPACES = bool(os.getenv("SPACE_ID"))
REQUIRE_LOGIN = os.getenv("REQUIRE_LOGIN", "true" if ON_SPACES and TEXT_PROVIDER == "hf" else "false").lower() == "true"

# When visitors use the owner's token (no login), cap how many chapters each one can write per day.
# 0 disables the cap; it's off by default for local use.
CHAPTERS_PER_DAY = int(os.getenv("CHAPTERS_PER_DAY", "10" if ON_SPACES and not REQUIRE_LOGIN else "0"))

# Generated images, audio and zips older than this are deleted.
FILE_RETENTION_HOURS = float(os.getenv("FILE_RETENTION_HOURS", "24"))

# Input limits for a public demo. Enforced on the server, not just in the browser, because the
# app's API can be called directly.
MAX_PROMPT_CHARS = 500
MAX_LORE_FILES = 5
MAX_LORE_FILE_MB = 5
# Caps embedding work: 20k words is ~200 chunks, a few seconds to index even on a small CPU.
MAX_LORE_WORDS = 20_000
