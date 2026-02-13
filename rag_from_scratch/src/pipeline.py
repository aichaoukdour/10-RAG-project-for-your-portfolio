import logging
from typing import Dict, List, Any

import pandas as pd

from embedding import Embedder
from vector_store import VectorStore
from retriever import Retriever
from generator import BaseGenerator, Generator, GenerationError, LocalAdvisor
from config import DEFAULT_TOP_K, DEFAULT_LLM_MODEL

logger = logging.getLogger(__name__)


class RAGPipeline:
   
    def __init__(
        self, 
        embedder: Embedder, 
        vector_store: VectorStore, 
        data: pd.DataFrame,
        model: str = DEFAULT_LLM_MODEL
    ) -> None:

        self.retriever = Retriever(embedder, vector_store, data)
        self.generator = Generator(model=model)
        self.fallback = LocalAdvisor()
        self.model = model
        
        logger.info(f"RAGPipeline initialized with model: {model}")

    def run(
        self, 
        query: str, 
        k: int = DEFAULT_TOP_K,
        use_fallback: bool = True
    ) -> Dict[str, Any]:

        logger.info("=" * 50)
        logger.info("Pipeline Execution Started")
        logger.info(f"Query: {query}")

        # 1. Retrieval Phase
        logger.info(f"Retrieving top {k} context chunks...")
        retrieval_results = self.retriever.search(query, k=k)
        
        if not retrieval_results:
            logger.warning("No relevant documents found")
            return {
                "query": query,
                "answer": "I couldn't find any relevant information in the knowledge base.",
                "context": [],
                "scores": [],
                "source": "no_results"
            }
        
        context_chunks = [res['text'] for res in retrieval_results]
        scores = [res['score'] for res in retrieval_results]

        # Log retrieved chunks for transparency
        for i, res in enumerate(retrieval_results):
            logger.debug(f"Chunk {i+1} (Score: {res['score']:.4f}): {res['text'][:80]}...")

        # 2. Generation Phase (with exception-based fallback)
        logger.info("Generating grounded answer...")
        source = "llm"
        try:
            answer = self.generator.generate_answer(query, context_chunks)
        except GenerationError as e:
            if use_fallback:
                logger.warning(f"LLM failed: {e}. Switching to LocalAdvisor.")
                answer = self.fallback.generate_answer(query, context_chunks)
                source = "fallback"
            else:
                raise

        logger.info(f"Answer generated via: {source}")
        logger.info("Pipeline Execution Finished")
        logger.info("=" * 50)

        return {
            "query": query,
            "answer": answer,
            "context": context_chunks,
            "scores": scores,
            "source": source
        }

    def get_salary_insight(
        self, 
        job_title: str,
        k: int = 5
    ) -> Dict[str, Any]:

        logger.info(f"Generating Salary Insight Report for: {job_title}")
        
        # Retrieve relevant salary data (retriever only — no LLM call here)
        query = f"What is the average salary and common remote work status for {job_title}?"
        retrieval_results = self.retriever.search(query, k=k)
        context_chunks = [res['text'] for res in retrieval_results]

        if not context_chunks:
            return {
                "job_title": job_title,
                "report": "No relevant salary data found.",
                "supporting_data": [],
                "num_records_analyzed": 0
            }

        # Generate a structured insight report
        insight_prompt = f"""Summarize the following data into a high-level Career Insight Report for: {job_title}

Include:
1. Typical Salary Range (min, max, average if apparent)
2. Most common location/remote status patterns
3. Experience levels represented
4. A practical tip for candidates applying for this role

Format your response clearly with bullet points or sections.

CONTEXT:
{chr(10).join(context_chunks)}
"""
        
        try:
            report = self.generator.generate_answer(insight_prompt, context_chunks)
        except GenerationError:
            logger.warning("Falling back to LocalAdvisor for insight report")
            report = self._generate_local_insight(job_title, context_chunks)
        
        return {
            "job_title": job_title,
            "report": report,
            "supporting_data": context_chunks,
            "num_records_analyzed": len(context_chunks)
        }

    def _generate_local_insight(
        self, 
        job_title: str, 
        context_chunks: List[str]
    ) -> str:
        """Generate a simple insight report locally when LLM is unavailable."""
        report = f"## Career Insight Report: {job_title}\n\n"
        report += "### Salary Data Found:\n"
        
        for chunk in context_chunks[:5]:
            report += f"• {chunk}\n"
        
        report += "\n### Summary:\n"
        report += f"Based on {len(context_chunks)} relevant records found in the database.\n"
        report += "\n[Note: Detailed analysis requires LLM access. This is a simplified local report.]"
        
        return report


if __name__ == "__main__":
    # Pipeline requires initialized components - see main.py for full example
    print("RAGPipeline module loaded successfully")
    print("Use main.py to run the interactive application")
