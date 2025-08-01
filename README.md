# Mood Story RAG

A Retrieval-Augmented Generation (RAG) system for extracting and interacting with stories. This repository contains the tools to build a generative loop with FAISS and LLMs.

## Project Structure

```
mood-story-rag/
├── app/
│   ├── main.py          # Streamlit UI
│   ├── rag_pipeline.py  # Core RAG logic
│   └── utils.py
├── data/
│   └── sample_lore/     # Some sample .txt/.pdf files
├── notebooks/
│   └── exploration.ipynb  # Kaggle exploratory notebook
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Setup

1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and configure your API tokens (e.g., Hugging Face access token).
4. Run the application:
   ```bash
   streamlit run app/main.py
   ```
