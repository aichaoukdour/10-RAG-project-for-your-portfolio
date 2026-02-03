# src/llm_client.py

import requests

# Change this to the port where Ollama is actually running
LLM_API_URL = "http://localhost:11434/api/generate"

def query_llm(prompt: str, max_tokens: int = 512, temperature: float = 0.3) -> str:
    """
    Send a prompt to the local Ollama API and get the response
    """
    payload = {
        "model": "llama3:latest",
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature
        }
    }
    response = requests.post(LLM_API_URL, json=payload)
    response.raise_for_status()
    data = response.json()
    return data.get('response', '')

