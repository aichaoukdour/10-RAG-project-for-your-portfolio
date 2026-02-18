from langchain_ollama import OllamaLLM
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from .config import LLM_MODEL

def get_assistant_chain():
    """
    Builds and returns the LangChain LCEL chain for the real-time assistant.
    """
    # 1. Initialize logic components
    llm = OllamaLLM(model=LLM_MODEL)
    search = DuckDuckGoSearchRun()
    
    # 2. Define the prompt template
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful AI assistant. You must answer the user's question 
        based *only* on the following search results. If the search results 
        are empty or do not contain the answer, say 'I could not find 
        any information on that.'

        Search Results:
        {context}

        Question:
        {question}
        """
    )
    
    # 3. Assemble the chain using LCEL
    # The chain takes a dictionary {"question": "..."}
    # It adds a "context" key by running the search tool
    # Then pipes the result into the prompt and finally the LLM
    chain = (
        RunnablePassthrough.assign(
            context=lambda x: search.run(x["question"])
        )
        | prompt
        | llm
    )
    
    return chain
