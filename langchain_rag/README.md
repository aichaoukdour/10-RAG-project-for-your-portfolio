# LangChain RAG Pipeline (Project 6)

This project demonstrates two **Retrieval-Augmented Generation** approaches using LangChain, inspired by the [official LangChain RAG tutorial](https://python.langchain.com/docs/tutorials/rag/).

All models run **locally via HuggingFace** — no API keys required.

## Features

| Approach | Description | When to Use |
|----------|-------------|-------------|
| **RAG Agent** | Intent-routing agent that decides whether to search or answer directly | General-purpose queries, mixed intent |
| **RAG Chain** | Always-retrieve LCEL chain (single LLM call per query) | Fast, focused Q&A over indexed content |

- **Indexing**: Fetches the Lilian Weng "LLM Powered Autonomous Agents" blog post, splits it into chunks, and stores embeddings in an `InMemoryVectorStore`.
- **Local Inference**: `google/flan-t5-base` via HuggingFace Transformers.
- **Semantic Search**: `all-MiniLM-L6-v2` sentence embeddings.
- **LangChain LCEL**: The chain uses LangChain Expression Language (`RunnableSequence`) for composable retrieval + generation.

## Architecture

```
Blog Post (HTML)
    │
    ▼
┌───────────────┐
│  WebBaseLoader │ ── bs4 SoupStrainer
└───────┬───────┘
        │
        ▼
┌──────────────────────────┐
│ RecursiveCharacterSplitter│ ── 1000 chars, 200 overlap
└───────┬──────────────────┘
        │
        ▼
┌──────────────────────────┐
│  InMemoryVectorStore     │ ── HuggingFace Embeddings
└───────┬──────────────────┘
        │
   ┌────┴─────┐
   ▼          ▼
┌──────┐  ┌──────┐
│ Agent│  │ Chain│
│(route)  │(LCEL)│
└──┬───┘  └──┬───┘
   │         │
   ▼         ▼
┌──────────────┐
│ flan-t5-base │
└──────────────┘
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application from the `langchain_rag/` directory:
   ```bash
   python src/main.py
   ```

3. Select **option 1** to index the blog post, then **option 2** (Agent) or **option 3** (Chain) to ask questions.

## Testing

```bash
pytest tests/ -v
```

## Example Usage

```
Select an option: 1
📚 Starting Indexing Pipeline
📄 Loaded 1 document(s), total characters: 43131
✂️  Split into 66 sub-documents.
📦 Stored 66 embeddings in vector store.
✅ Index built.

Select an option: 2
🗣️  Your question: What is task decomposition?

🕵️  Agent decided to SEARCH for: 'What is task decomposition?'
📎 Retrieved 3 documents.

💬 Answer:
Task decomposition is a technique that breaks down complex tasks
into smaller, manageable steps...
```
