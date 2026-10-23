"""Host the app on Modal: a CPU container that runs only while someone is using it.

Modal's free Starter plan includes $30 of compute a month; 4 CPU cores and 8 GB cost roughly
$0.25/hour, so that is over 100 hours of uptime, and the container shuts down after 5 idle minutes.
Text comes from Groq; the CPU draws SD-Turbo pictures and runs the lore embeddings. (A GPU would
draw in about a second, but Modal requires a payment method for GPUs.)

One-time setup (the secret holds only the Groq key, read from .env so it never appears on screen):
    modal setup
    f=$(mktemp) && grep '^GROQ_API_KEY=' .env > "$f" && modal secret create mood-story --from-dotenv "$f"; rm -f "$f"
Deploy, from the project root:
    modal deploy deploy/modal_app.py
"""
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
HF_CACHE = "/root/hf-cache"


def download_models():
    """Bake the models into the image, so a cold start loads them from disk instead of the internet."""
    from diffusers import AutoPipelineForText2Image
    from sentence_transformers import SentenceTransformer

    AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo")
    SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


image = (
    modal.Image.debian_slim(python_version="3.12")
    # The same locked versions as everywhere else, including CPU-only PyTorch.
    .pip_install_from_requirements(str(ROOT / "requirements.txt"))
    .env({"HF_HOME": HF_CACHE, "CHAPTERS_PER_DAY": "10"})
    .run_function(download_models)
    # Only the code and data the app needs: never .env or anything else in the working tree. The
    # app finds its data relative to the `app` package, so both go side by side under /root.
    .add_local_dir(ROOT / "app", "/root/app")
    .add_local_dir(ROOT / "data", "/root/data")
)

app = modal.App("mood-story", image=image)


@app.function(
    cpu=4,
    memory=8 * 1024,
    secrets=[modal.Secret.from_name("mood-story")],
    # Gradio keeps each visitor's session in memory, so every request must reach the same container.
    max_containers=1,
    scaledown_window=5 * 60,
    timeout=10 * 60,
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def web():
    import gradio as gr
    from fastapi import FastAPI

    from app.main import SERVER_OPTIONS, create_app, start_background_tasks

    start_background_tasks()
    return gr.mount_gradio_app(FastAPI(), create_app().queue(), path="/", **SERVER_OPTIONS)
