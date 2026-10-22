"""Shared test setup.

Tests never call real models or the network: model calls are replaced with small fakes, so the
suite runs in seconds, costs nothing, and checks our own logic rather than a model's mood.
"""
import os
import re
import zlib

import numpy as np
import pytest

# Must be set before `app` is imported: config reads the environment once at import time.
# Real environment variables take precedence over .env, so the developer's real token is never used.
os.environ.update({
    "MODEL_BACKEND": "api",
    "HF_TOKEN": "hf_test_token_not_real",
    "GROQ_API_KEY": "",  # empty, so the defaults are the HF provider; Groq tests switch it on
    "REQUIRE_LOGIN": "false",
    "CHAPTERS_PER_DAY": "0",
})


def fake_embed(texts):
    """Deterministic bag-of-words vectors: texts sharing words point the same way.

    Stands in for the MiniLM embedder so retrieval tests need no model download.
    """
    vectors = np.zeros((len(texts), 256), dtype="float32")
    for row, text in enumerate(texts):
        for word in re.findall(r"[a-z]+", text.lower()):
            vectors[row, zlib.crc32(word.encode()) % 256] += 1
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


@pytest.fixture(autouse=True)
def offline_embeddings(monkeypatch):
    import app.rag_pipeline

    monkeypatch.setattr(app.rag_pipeline, "_embed", fake_embed)


@pytest.fixture
def output_dir(tmp_path, monkeypatch):
    """Send generated files (images, zips) to a temporary folder."""
    import app.main
    import app.multimodal

    monkeypatch.setattr(app.main, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(app.multimodal, "OUTPUT_DIR", tmp_path)
    return tmp_path
