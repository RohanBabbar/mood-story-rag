import os
from PyPDF2 import PdfReader

def load_custom_lore(file):
    """Load text from PDF or TXT file."""
    if file is None:
        return "[No lore uploaded]"
    try:
        if file.name.endswith('.pdf'):
            reader = PdfReader(file)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
            if not text.strip():
                return "[Error: PDF has no extractable text]"
            return text
        elif file.name.endswith('.txt'):
            with open(file.name, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            return "[Error: Unsupported file type]"
    except Exception as e:
        return f"[Error reading file: {e}]"

def chunk_text(text, chunk_size=100):
    """Split text into chunks for FAISS indexing."""
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
