# GraphRAG with Ollama & Docker

This project implements a Graph Retrieval-Augmented Generation (GraphRAG) pipeline using **Ollama** (Mistral) and **NetworkX**. It is containerized using Docker for easy deployment.

## 🚀 Getting Started

### Prerequisites
- Docker & Docker Compose
- (Optional) NVIDIA GPU with Container Toolkit for faster inference

### Installation & Running

1.  **Build and Run**:
    ```bash
    docker-compose up --build
    ```
    *Note: The first run will take time as it pulls the Ollama image.*

2.  **Pull the Model**:
    Once the containers are running, you need to ensure the `mistral` model is available in the Ollama container.
    Open a separate terminal:
    ```bash
    docker-compose exec ollama ollama pull mistral
    ```
    *Wait for the download to complete.*

3.  **Run the GraphRAG App**:
    You can interact with the app via the command line inside the container:
    ```bash
    docker-compose run --rm app
    ```

### Usage

1.  **Ingest Data**: select option 1 and paste text.
    *Example Text*:
    > OpenAI was founded by Sam Altman and Elon Musk. OpenAI developed GPT-4. GPT-4 powers ChatGPT. Microsoft partnered with OpenAI. Microsoft invested 10 billion dollars in OpenAI. ChatGPT is used by millions of users worldwide.

2.  **Ask Question**: Select option 2.
    *Example*: "Which company invested in the company that built ChatGPT?"

3.  **Graph Stats**: See how many nodes and edges are in your knowledge graph.

## 🛠️ Tech Stack
-   **LLM**: Mistral (via Ollama)
-   **Orchestration**: LangChain
-   **Graph**: NetworkX
-   **Containerization**: Docker
