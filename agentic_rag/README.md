# Agentic RAG Pipeline (Project 5)

This project demonstrates an **Agentic RAG** pattern where a router agent decides whether to query a knowledge base or respond directly using a local LLM.

## Features
- **Agentic Routing**: Automatically detects if a query needs document context based on intent analysis.
- **Local Inference**: Uses `google/flan-t5-base` via HuggingFace for fully local, private execution.
- **Semantic Search**: Powered by ChromaDB and `all-MiniLM-L6-v2` embeddings.
- **Privacy First**: No API keys required; all data stays on your machine.

## Setup
1. Place PDF documents into the `agentic_rag/data/` directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python src/main.py
   ```

## Testing
Run the unit tests to verify the routing logic:
```bash
pytest tests/test_controller.py
```

## How it Works
1. **Controller**: Analyzes the query for keywords like "PDF", "document", or "summarize".
2. **Search Mode**: If keywords are found, the system retrieves relevant chunks from ChromaDB.
3. **Direct Mode**: If no keywords are found, the query is sent directly to the LLM for a general response.
