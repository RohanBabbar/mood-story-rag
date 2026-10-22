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

# "api"   -> Hugging Face Inference Providers (no GPU needed, used for deployment)
# "local" -> run LLaMA + Stable Diffusion on this machine (needs requirements-local.txt)
BACKEND = os.getenv("MODEL_BACKEND", "api").lower()

TEXT_MODEL = os.getenv(
    "TEXT_MODEL",
    "meta-llama/Llama-3.1-8B-Instruct" if BACKEND == "api" else "meta-llama/Llama-3.2-3B-Instruct",
)
IMAGE_MODEL = os.getenv(
    "IMAGE_MODEL",
    "black-forest-labs/FLUX.1-schnell" if BACKEND == "api" else "stable-diffusion-v1-5/stable-diffusion-v1-5",
)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

OUTPUT_DIR = Path(tempfile.gettempdir()) / "mood-story"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Backups tried in order when the main model is unavailable (removed, overloaded or unsupported).
# Set to an empty string to disable.
TEXT_FALLBACK_MODELS = [m for m in os.getenv("TEXT_FALLBACK_MODELS", "meta-llama/Llama-3.3-70B-Instruct").split(",") if m]
IMAGE_FALLBACK_MODELS = [m for m in os.getenv("IMAGE_FALLBACK_MODELS", "stabilityai/stable-diffusion-xl-base-1.0").split(",") if m]

# Hugging Face Spaces sets SPACE_ID. On a Space, visitors sign in with Hugging Face by default so
# inference is billed to their own account instead of the Space owner's.
ON_SPACES = bool(os.getenv("SPACE_ID"))
REQUIRE_LOGIN = os.getenv("REQUIRE_LOGIN", "true" if ON_SPACES else "false").lower() == "true"

# When visitors use the owner's token (no login), cap how many chapters each one can write per day.
# 0 disables the cap; it's off by default for local use.
CHAPTERS_PER_DAY = int(os.getenv("CHAPTERS_PER_DAY", "10" if ON_SPACES and not REQUIRE_LOGIN else "0"))

# Generated images, audio and zips older than this are deleted.
FILE_RETENTION_HOURS = float(os.getenv("FILE_RETENTION_HOURS", "24"))
