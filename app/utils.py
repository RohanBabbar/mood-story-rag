import time
from pathlib import Path

from pypdf import PdfReader

SUPPORTED_LORE_TYPES = [".pdf", ".txt", ".md"]


def load_custom_lore(path):
    """Read text from a PDF, TXT or Markdown file. Raises ValueError on unusable files."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif suffix in (".txt", ".md"):
        text = path.read_text(encoding="utf-8", errors="replace")
    else:
        raise ValueError(f"{path.name}: unsupported file type (use PDF, TXT or MD)")

    if not text.strip():
        raise ValueError(f"{path.name}: no readable text found (scanned PDFs aren't supported)")
    return text


def chunk_text(text, chunk_size=120, overlap=30):
    """Split text into overlapping word windows so facts aren't cut in half at chunk edges."""
    words = text.split()
    step = chunk_size - overlap
    return [" ".join(words[i:i + chunk_size]) for i in range(0, max(len(words) - overlap, 1), step)]


def delete_old_files(directory, max_age_hours):
    """Remove files older than max_age_hours so a long-running server doesn't fill its disk."""
    cutoff = time.time() - max_age_hours * 3600
    for path in Path(directory).iterdir():
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            pass  # another request may have removed it already
