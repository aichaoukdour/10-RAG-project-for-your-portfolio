# RAG from Scratch

A complete Retrieval-Augmented Generation (RAG) system built from scratch using Python. This project demonstrates how to build a professional-grade RAG pipeline for a "Career Advisor" application that answers salary and career questions.

## 🎯 Features

- **Semantic Search**: Uses sentence-transformers for high-quality embeddings
- **FAISS Vector Store**: Efficient similarity search with L2-normalized vectors
- **OpenAI Integration**: GPT-powered answer generation with context grounding
- **Fallback Mechanism**: LocalAdvisor for offline/no-credit scenarios
- **Interactive CLI**: User-friendly command-line interface
- **Modular Architecture**: Clean separation of concerns for easy extension

## 📁 Project Structure

```
rag_from_scratch/
├── src/
│   ├── __init__.py       # Package exports
│   ├── config.py         # Centralized configuration
│   ├── main.py           # CLI entry point
│   ├── pipeline.py       # RAG orchestration
│   ├── embedding.py      # Text to vectors
│   ├── vector_store.py   # FAISS index management
│   ├── retriever.py      # Semantic search
│   ├── generator.py      # LLM answer generation
│   └── ingestion.py      # Data loading & cleaning
├── data/
│   ├── raw/              # Source CSV files
│   └── processed/        # Cleaned data & FAISS index
├── tests/                # Pytest test suite
├── notebooks/            # Exploration notebooks
├── requirements.txt      # Dependencies
└── .env.example          # Environment template
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd rag_from_scratch
pip install -r requirements.txt
```

### 2. Configure API Key (Optional)

Create a `.env` file from the template:

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

> **Note**: The system works without an API key using the LocalAdvisor fallback!

### 3. Add Your Data

Place your salary data CSV in `data/raw/salaries.csv` with columns:
- `work_year`, `experience_level`, `employment_type`, `job_title`
- `salary_in_usd`, `employee_residence`, `remote_ratio`
- `company_location`, `company_size`

### 4. Run the Application

```bash
cd src
python main.py
```

## 🧪 Running Tests

```bash
# From project root
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│   Query     │────▶│   Embedder   │────▶│   Retriever  │
└─────────────┘     └──────────────┘     └──────────────┘
                                                │
                                                ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│   Answer    │◀────│  Generator   │◀────│ VectorStore  │
└─────────────┘     │  (or Local)  │     │   (FAISS)    │
                    └──────────────┘     └──────────────┘
```

### Components

| Component | Description |
|-----------|-------------|
| `Embedder` | Converts text to 384-dim vectors using `all-MiniLM-L6-v2` |
| `VectorStore` | FAISS IndexFlatIP for cosine similarity search |
| `Retriever` | Combines embedding + search for semantic retrieval |
| `Generator` | OpenAI GPT for grounded answer generation |
| `LocalAdvisor` | Rule-based fallback when LLM is unavailable |
| `RAGPipeline` | Orchestrates the full flow with error handling |

## 📦 Usage as a Library

```python
from src import RAGPipeline, Embedder, VectorStore
import pandas as pd

# Load your data
df = pd.read_csv("data/processed/cleaned_salaries.csv")

# Initialize components
embedder = Embedder()
store = VectorStore(dimension=384)
store.load("data/processed/faiss_index.bin")

# Create pipeline
pipeline = RAGPipeline(embedder, store, df)

# Query
result = pipeline.run("What is the salary for a Data Scientist?")
print(result['answer'])
```

## 🔧 Configuration

All settings are centralized in `src/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `EMBEDDING_DIMENSION` | `384` | Vector dimensions |
| `DEFAULT_LLM_MODEL` | `gpt-4o-mini` | OpenAI model for generation |
| `DEFAULT_TOP_K` | `5` | Number of chunks to retrieve |

## 📝 License

MIT License - feel free to use this for learning and projects!
