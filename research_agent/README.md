# AI Research Agent (Project 9)

This project automates the first round of web research by taking a query, searching the web, extracting relevant content, and generating an extractive summary using semantic embeddings.

Inspired by the guide: "Build an AI Agent to Automate Your Research" by Aman Kharwal.

## Features

- **Automated Search**: Uses DuckDuckGo to find the most relevant URLs for a query.
- **Web Scraping**: Fetches and cleans text from web pages using `BeautifulSoup`.
- **Semantic Ranking**: Uses `sentence-transformers/all-MiniLM-L6-v2` to rank passages by meaning rather than keywords.
- **Extractive Summary**: Generates a concise summary by selecting the most relevant sentences across multiple sources.
- **Local Inference**: All embeddings and ranking logic run locally.

## Architecture

1.  **Search**: `duckduckgo_search` finds top results.
2.  **Fetch**: `requests` + `bs4` scrapes text, removing noise (scripts, headers, etc.).
3.  **Chunk**: Text is split into small passages (~120 words).
4.  **Embed**: Query and passages are converted to vectors.
5.  **Rank**: Cosine similarity finds the most relevant passages.
6.  **Summarize**: Sentences within top passages are ranked and selected for the final summary.

## Setup

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2.  Run the agent:
    ```bash
    python src/main.py
    ```

## Example

**Query**: "What causes urban heat islands and how can cities reduce them?"

**Output**:
- A list of the most relevant passages with similarity scores.
- A 3-sentence extractive summary with cited sources.

## Testing

Run tests using pytest:
```bash
pytest tests/
```
