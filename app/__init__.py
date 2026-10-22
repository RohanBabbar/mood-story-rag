import os
import sys

# On macOS, faiss and PyTorch each bundle their own OpenMP runtime; running both
# multi-threaded segfaults. This must be set before either library is imported.
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
