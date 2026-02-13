import logging
import sys
import json
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from config import OLLAMA_BASE_URL, LLM_MODEL, LLM_TEMPERATURE
from extractor import GraphExtractor
from knowledge_graph import KnowledgeGraph, kg
from retriever import retrieve_graph_context

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 3. Enterprise Knowledge Example (Data Source)
# ──────────────────────────────────────────────
SAMPLE_TEXT = """
OpenAI was founded by Sam Altman and Elon Musk.
OpenAI developed GPT-4.
GPT-4 powers ChatGPT.
Microsoft partnered with OpenAI.
Microsoft invested 10 billion dollars in OpenAI.
ChatGPT is used by millions of users worldwide.
"""


class GraphRAGPipeline:
    def __init__(self):
        self.extractor = GraphExtractor()
        self.graph = KnowledgeGraph()
        self.llm = ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE
        )

        self.answer_prompt = PromptTemplate(
            template="""
            Answer the question using ONLY the context below.

            Context:
            {context}

            Question:
            {question}

            Answer:
            """,
            input_variables=["context", "question"]
        )
        self.answer_chain = self.answer_prompt | self.llm

    # ──────────────────────────────────────────
    # Step 3: Extract triples from text
    # ──────────────────────────────────────────
    def ingest(self, text: str):
        """Extracts triples from text and builds the graph."""
        print("\n📌 Step 3: Extracting knowledge graph triples...\n")
        triples = self.extractor.extract(text)
        print(json.dumps(triples, indent=2))
        print(f"\n✅ Extracted {len(triples)} triples.")

        # ──────────────────────────────────────
        # Step 4: Build the graph from triples
        # ──────────────────────────────────────
        print("\n📌 Step 4: Building the Knowledge Graph...\n")
        self.graph.add_triples(triples)
        stats = self.graph.get_stats()
        print(f"   Nodes: {stats['nodes']}")
        print(f"   Edges: {stats['edges']}")
        print("✅ Knowledge Graph built successfully.")
        return triples

    # ──────────────────────────────────────────
    # Step 5: Multi-Hop Retrieval + Answer
    # ──────────────────────────────────────────
    def ask(self, question: str):
        """Retrieves multi-hop context and generates an answer."""
        logger.info(f"Querying: {question}")

        # Find entities from the question that exist in the graph
        entities_in_graph = [
            n for n in self.graph.graph.nodes
            if n.lower() in question.lower()
        ]

        if not entities_in_graph:
            print("❌ No matching entities found in the graph for this question.")
            return

        print(f"\n🔍 Entities found: {entities_in_graph}")

        # Multi-hop retrieval for each matched entity
        context_parts = []
        for entity in entities_in_graph:
            print(f"\n📌 Step 5: Multi-Hop Retrieval for '{entity}'...")
            ctx = retrieve_graph_context(entity, max_depth=2)
            if ctx:
                context_parts.append(ctx)
                print(f"   Context: {ctx}")

        full_context = ". ".join(context_parts)

        if not full_context:
            print("❌ No relevant context found in the graph.")
            return

        # Generate answer using the LLM
        print("\n📌 Step 6: Generating answer with LLM...\n")
        response = self.answer_chain.invoke({
            "context": full_context,
            "question": question
        })

        print("┌─────────── Answer ───────────┐")
        print(f"  {response.content}")
        print("└──────────────────────────────┘\n")


def main():
    pipeline = GraphRAGPipeline()

    print("=" * 50)
    print("  GraphRAG — Multi-Hop Retrieval (Ollama + Docker)")
    print("=" * 50)
    print("Type 'exit' to quit.\n")

    while True:
        print("\nOptions:")
        print("  1. Ingest custom text")
        print("  2. Ask a question")
        print("  3. Show graph stats")
        print("  4. Load demo data (AI industry)")
        choice = input("\nSelect: ").strip()

        if choice == "1":
            print("Enter text (end with an empty line):")
            lines = []
            while True:
                line = input()
                if not line:
                    break
                lines.append(line)
            text = "\n".join(lines)
            if text:
                pipeline.ingest(text)

        elif choice == "2":
            q = input("Question: ").strip()
            if q:
                pipeline.ask(q)

        elif choice == "3":
            stats = pipeline.graph.get_stats()
            print(f"\n📊 Graph Stats — Nodes: {stats['nodes']}, Edges: {stats['edges']}")
            if kg.number_of_nodes() > 0:
                print(f"   All entities: {list(kg.nodes)}")

        elif choice == "4":
            print("\n📄 Loading demo data (AI industry)...")
            print(f"   Text: {SAMPLE_TEXT.strip()}")
            pipeline.ingest(SAMPLE_TEXT)

        elif choice.lower() == "exit":
            print("Goodbye!")
            break


if __name__ == "__main__":
    main()
