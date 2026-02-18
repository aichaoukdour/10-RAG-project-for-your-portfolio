# Multimodal RAG Pipeline (Project 7)

This project demonstrates a **Multimodal RAG** pipeline that processes PDF documents containing text, tables, and images. Adapted from the [IBM Granite Multimodal RAG tutorial](https://www.ibm.com/granite/docs/tutorials/multimodal-rag/) for **fully local execution** with HuggingFace models.

## Features

| Feature | Technology |
|---------|-----------|
| **PDF Parsing** | Docling (text, tables, images) |
| **Image Captioning** | `Salesforce/blip-image-captioning-base` |
| **Embeddings** | `all-MiniLM-L6-v2` |
| **Vector Store** | Milvus Lite (local file DB) |
| **LLM** | `google/flan-t5-base` |
| **Orchestration** | LangChain LCEL chain |

- **Truly Multimodal**: Indexes text chunks, tables (as markdown), and image descriptions (BLIP captions)
- **Local & Private**: No API keys required — all models run on your machine
- **PDF Processing**: Docling extracts structured content from PDFs including OCR-free text and embedded images

## Architecture

```
PDF Document
    │
    ▼
┌──────────────────┐
│  Docling Parser   │
└──┬──────┬──────┬─┘
   │      │      │
   ▼      ▼      ▼
 Text   Tables  Images
chunks  (md)    (PIL)
   │      │      │
   │      │      ▼
   │      │  ┌──────────┐
   │      │  │ BLIP      │ → captions
   │      │  └──────────┘
   │      │      │
   ▼      ▼      ▼
┌──────────────────────────┐
│  Milvus Lite Vector DB   │ ← HuggingFace Embeddings
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  LangChain RAG Chain     │ → flan-t5-base → Answer
└──────────────────────────┘
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application from the `multimodal_rag/` directory:
   ```bash
   python src/main.py
   ```

3. Select **option 1** to download & process the sample PDF, then **option 3** to ask questions.

## Testing

```bash
pytest tests/ -v
```

## Example Usage

```
Select an option: 1
⬇️  Downloading sample PDF...
📖 Converting PDF: data/AR_2020_WEB2.pdf
✂️  Created 45 text chunks.
📊 Extracted 3 tables.
🖼️  Captioning 8 images...
📦 Added 56 documents to the vector database.
✅ Document indexed successfully!

Select an option: 3
🗣️  Your question: How much food was distributed?

🔍 Querying: 'How much food was distributed?'

💬 Answer:
The Midwest Food Bank distributed over 292 million pounds of food...
```
