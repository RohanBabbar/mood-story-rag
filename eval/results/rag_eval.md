# Retrieval evaluation

30 questions over 2 lore files (1588 words). A hit means the answering phrase is in the top-k chunks. "Lore retrieved @4" is how much of the whole lore the top 4 chunks cover: recall is only meaningful when that share is small. With 30 questions each one is ~3 points, so gaps under ~10 points are within noise.

| Embedder | Chunk words | Overlap | Chunks | Recall@1 | Recall@2 | Recall@4 | Lore retrieved @4 |
|---|---|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | 60 | 0 | 28 | 47% | 50% | 70% | 14% |
| all-MiniLM-L6-v2 | 60 | 20 | 39 | 47% | 73% | 80% | 10% |
| all-MiniLM-L6-v2 | 120 | 0 | 15 | 40% | 63% | 80% | 27% |
| all-MiniLM-L6-v2 **(app)** | 120 | 30 | 18 | 57% | 73% | 83% | 22% |
| all-MiniLM-L6-v2 | 200 | 0 | 9 | 70% | 83% | 87% | 44% |
| all-MiniLM-L6-v2 | 200 | 50 | 11 | 80% | 87% | 97% | 36% |
| bge-small-en-v1.5 | 60 | 0 | 28 | 43% | 53% | 70% | 14% |
| bge-small-en-v1.5 | 60 | 20 | 39 | 57% | 70% | 80% | 10% |
| bge-small-en-v1.5 | 120 | 0 | 15 | 57% | 70% | 77% | 27% |
| bge-small-en-v1.5 | 120 | 30 | 18 | 60% | 73% | 93% | 22% |
| bge-small-en-v1.5 | 200 | 0 | 9 | 53% | 67% | 83% | 44% |
| bge-small-en-v1.5 | 200 | 50 | 11 | 70% | 83% | 93% | 36% |
