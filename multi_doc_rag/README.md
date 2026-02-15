# Multi-Document RAG System (Project 4)

This project allows you to query multiple PDF documents using a local RAG pipeline powered by LangChain, ChromaDB, and Ollama.

## Features
- **Multi-Document Support**: Ingest multiple PDFs from a folder.
- **Local Inference**: Uses Ollama (Llama 3/Mistral) for privacy and zero cost.
- **Semantic Search**: Powered by HuggingFace embeddings (`all-MiniLM-L6-v2`).
- **Persistent Storage**: Uses ChromaDB to save vector indices locally.

## Prerequisities
- Python 3.11+
- [Ollama](https://ollama.com/) running with `llama3` pull: `ollama pull llama3`
- Docker (optional, for containerized execution)

## Setup
1. Place your PDF files in the `multi_doc_rag/data/` directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python src/main.py
   ```

## Docker Execution
Run with Docker Compose:
```bash
docker-compose up --build
```

## Testing
Run the unit tests using `pytest`:
```bash
pytest multi_doc_rag/tests
```
The tests are designed to be fast and use mocks for heavy dependencies like ChromaDB and Ollama.

## Architecture
- **Loader**: `PyPDFLoader` for efficient PDF text extraction.
- **Chunker**: `RecursiveCharacterTextSplitter` with context overlap.
- **Embeddings**: `HuggingFaceEmbeddings`.
- **Vector Store**: `Chroma`.
- **Pipeline**: LangChain `LCEL` chain.
