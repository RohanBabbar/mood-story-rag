from dataclasses import dataclass, field
from functools import lru_cache

import faiss
import numpy as np

from app.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embedder():
    """Load the sentence embedder once, on first use."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")


def _embed(texts):
    vectors = get_embedder().encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return np.asarray(vectors, dtype="float32")


@dataclass
class LoreIndex:
    chunks: list
    sources: list = field(default_factory=list)
    index: faiss.Index = None


def build_faiss_index(chunks, sources=None):
    """Embed lore chunks into a cosine-similarity FAISS index."""
    embeddings = _embed(chunks)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return LoreIndex(chunks=chunks, sources=sources or [], index=index)


def retrieve_context(query, lore, k=4):
    """Return the top-k lore chunks most relevant to the query."""
    if lore is None or not lore.chunks:
        return ""
    k = min(k, len(lore.chunks))
    _, indices = lore.index.search(_embed([query]), k)
    # FAISS pads with -1 when it finds fewer than k results.
    return "\n\n".join(lore.chunks[i] for i in indices[0] if i >= 0)
