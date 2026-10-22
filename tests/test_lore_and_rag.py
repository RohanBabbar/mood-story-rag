import os
import time

import pytest

from app.rag_pipeline import build_faiss_index, retrieve_context
from app.utils import chunk_text, delete_old_files, load_custom_lore


def test_short_text_is_a_single_chunk():
    assert chunk_text("just a few words here") == ["just a few words here"]


def test_chunks_overlap_so_facts_are_not_cut_in_half():
    words = [f"w{i}" for i in range(300)]
    chunks = chunk_text(" ".join(words), chunk_size=120, overlap=30)
    first, second = chunks[0].split(), chunks[1].split()
    assert first[-30:] == second[:30]
    # Every word ends up in at least one chunk.
    assert set(words) <= {w for c in chunks for w in c.split()}


def test_loads_text_and_markdown(tmp_path):
    for name in ("world.txt", "world.md"):
        path = tmp_path / name
        path.write_text("The queen cannot lie.", encoding="utf-8")
        assert load_custom_lore(path) == "The queen cannot lie."


def test_rejects_unsupported_and_empty_files(tmp_path):
    docx = tmp_path / "notes.docx"
    docx.write_bytes(b"irrelevant")
    with pytest.raises(ValueError, match="unsupported"):
        load_custom_lore(docx)

    empty = tmp_path / "empty.txt"
    empty.write_text("   \n", encoding="utf-8")
    with pytest.raises(ValueError, match="no readable text"):
        load_custom_lore(empty)


def test_retrieval_returns_the_relevant_passage():
    lore = build_faiss_index([
        "The stone wyrm Kessath guards the Hollow Caves beneath the mountain.",
        "Queen Isolde rules the kingdom of Veyra and cannot tell a lie.",
        "Bakers in the harbour town sell honey bread at dawn.",
    ])
    assert "Kessath" in retrieve_context("who guards the caves", lore, k=1)


def test_retrieval_with_fewer_chunks_than_k_has_no_duplicates():
    # Regression: FAISS pads missing results with -1, which used to repeat the last chunk.
    lore = build_faiss_index(["Only one passage of lore."])
    assert retrieve_context("anything", lore, k=4) == "Only one passage of lore."


def test_no_lore_means_no_context():
    assert retrieve_context("anything", None) == ""


def test_old_generated_files_are_deleted(tmp_path):
    old, new = tmp_path / "old.jpg", tmp_path / "new.jpg"
    old.write_text("x")
    new.write_text("x")
    a_day_and_more_ago = time.time() - 30 * 3600
    os.utime(old, (a_day_and_more_ago, a_day_and_more_ago))

    delete_old_files(tmp_path, max_age_hours=24)

    assert not old.exists()
    assert new.exists()
