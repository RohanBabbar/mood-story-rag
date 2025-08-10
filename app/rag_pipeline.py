import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize embedding model (runs on CPU or GPU automatically via sentence-transformers)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def build_faiss_index(chunks):
    """Create FAISS index from text chunks."""
    embeddings = embedder.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, chunks

def retrieve_context(prompt, index, chunks, k=3):
    """Retrieve top-k relevant lore chunks for a given prompt."""
    query_embedding = embedder.encode([prompt])
    distances, indices = index.search(np.array(query_embedding), k)
    return "\n".join([chunks[i] for i in indices[0]])
