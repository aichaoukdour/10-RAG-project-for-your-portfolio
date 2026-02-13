from typing import List, Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from config import OLLAMA_BASE_URL, LLM_MODEL, LLM_TEMPERATURE

class GraphExtractor:
    """Extracts entities and relationships from text using a local LLM."""

    def __init__(self):
        self.llm = ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE
        )
        self.parser = JsonOutputParser()
        self.prompt = PromptTemplate(
            template="""
            You are an expert knowledge graph builder.
            Extract entities and relationships from the text.
            Return ONLY a JSON list. Each item must contain:
            - "head": source entity
            - "relation": relationship
            - "tail": target entity

            Text:
            {text}

            Output JSON:
            """,
            input_variables=["text"],
        )
        self.chain = self.prompt | self.llm | self.parser

    def extract(self, text: str) -> List[Dict[str, str]]:
        """Extracts triples from the given text."""
        try:
            return self.chain.invoke({"text": text})
        except Exception as e:
            print(f"Error extracting triples: {e}")
            return []
