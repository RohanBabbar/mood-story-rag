"""Retrieval evaluation: does RAG put the right lore in front of the model?

For each question in rag_questions.jsonl we know the exact phrase from the lore that answers it.
A retrieval counts as a hit at k if that phrase appears in any of the top-k chunks. This is
measured for several chunk sizes, overlaps and embedding models, so the app's settings are
chosen from data rather than guessed.

Run from the project root:  python -m eval.rag_eval
"""
import json
import re
import time
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from app.utils import chunk_text, load_custom_lore

ROOT = Path(__file__).resolve().parent.parent
LORE_DIR = ROOT / "data" / "sample_lore"
QUESTIONS = Path(__file__).parent / "rag_questions.jsonl"
RESULTS = Path(__file__).parent / "results"

# (model id, prefix added to queries). bge models are trained to expect a search instruction.
EMBEDDERS = [
    ("sentence-transformers/all-MiniLM-L6-v2", ""),
    ("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: "),
]
CHUNKING = [(60, 0), (60, 20), (120, 0), (120, 30), (200, 0), (200, 50)]
KS = (1, 2, 4)
APP_SETTINGS = ("sentence-transformers/all-MiniLM-L6-v2", 120, 30)  # what the app uses today


def normalize(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def evaluate(model, prefix, questions, chunks):
    embed = lambda texts: model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    index = faiss.IndexFlatIP(model.get_embedding_dimension())
    index.add(embed(chunks))
    _, top = index.search(embed([prefix + q["question"] for q in questions]), max(KS))
    hits = {k: 0 for k in KS}
    misses = []
    for q, row in zip(questions, top, strict=True):
        found_at = next((rank for rank, i in enumerate(row) if normalize(q["fact"]) in normalize(chunks[i])), None)
        for k in KS:
            hits[k] += found_at is not None and found_at < k
        if found_at is None or found_at >= 4:
            misses.append(q["question"])
    return {f"recall@{k}": hits[k] / len(questions) for k in KS}, misses


def main():
    # Two regions with deliberately similar details (two queens, two guardians, two bakeries...),
    # so retrieval has to tell look-alike passages apart. Chunked per file, like the app.
    files = sorted(LORE_DIR.glob("*.md"))
    texts = [load_custom_lore(f) for f in files]
    lore = "\n".join(texts)
    questions = [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]
    missing = [q["fact"] for q in questions if normalize(q["fact"]) not in normalize(lore)]
    assert not missing, f"facts not found verbatim in the lore: {missing}"

    rows = []
    for model_id, prefix in EMBEDDERS:
        model = SentenceTransformer(model_id, device="cpu")
        for size, overlap in CHUNKING:
            chunks = [c for t in texts for c in chunk_text(t, chunk_size=size, overlap=overlap)]
            start = time.perf_counter()
            scores, misses = evaluate(model, prefix, questions, chunks)
            rows.append({"embedder": model_id.split("/")[-1], "chunk_words": size, "overlap": overlap,
                         "chunks": len(chunks), "share_retrieved@4": min(4 / len(chunks), 1), **scores, "seconds": round(time.perf_counter() - start, 2),
                         "is_app_setting": (model_id, size, overlap) == APP_SETTINGS, "misses@4": misses})

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "rag_eval.json").write_text(json.dumps(rows, indent=2))
    lines = [(f"# Retrieval evaluation\n\n{len(questions)} questions over {len(files)} lore files "
             f"({len(lore.split())} words). A hit means the answering phrase is in the top-k chunks. "
             "\"Lore retrieved @4\" is how much of the whole lore the top 4 chunks cover: recall is only "
             "meaningful when that share is small. With 30 questions each one is ~3 points, so gaps under "
             "~10 points are within noise.\n"),
             "| Embedder | Chunk words | Overlap | Chunks | Recall@1 | Recall@2 | Recall@4 | Lore retrieved @4 |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        mark = " **(app)**" if r["is_app_setting"] else ""
        lines.append(f"| {r['embedder']}{mark} | {r['chunk_words']} | {r['overlap']} | {r['chunks']} | "
                     f"{r['recall@1']:.0%} | {r['recall@2']:.0%} | {r['recall@4']:.0%} | {r['share_retrieved@4']:.0%} |")
    (RESULTS / "rag_eval.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    app = next(r for r in rows if r["is_app_setting"])
    print(f"\nApp setting misses at k=4: {app['misses@4'] or 'none'}")


if __name__ == "__main__":
    main()
