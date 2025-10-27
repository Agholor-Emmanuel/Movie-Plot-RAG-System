# utils/llm_utils.py
from typing import Dict, List
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from utils.prompt_utils import get_rag_prompt
from config import (
    GEMINI_MODEL_NAME, GEMINI_API_KEY,
    OPENAI_MODEL_NAME, OPENAI_API_KEY,
    CLAUDE_MODEL_NAME, ANTHROPIC_API_KEY,
    MAX_TOKENS
)

def create_gemini_llm():
    """Create and return a Google Generative AI LLM instance."""
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL_NAME,
        google_api_key=GEMINI_API_KEY,
        max_output_tokens=MAX_TOKENS
    )

def create_openai_llm():
    """Create and return an OpenAI LLM instance."""
    return ChatOpenAI(
        model=OPENAI_MODEL_NAME,
        openai_api_key=OPENAI_API_KEY,
        max_tokens=MAX_TOKENS
    )

def create_claude_llm():
    """Create and return a Claude LLM instance."""
    return ChatAnthropic(
        model=CLAUDE_MODEL_NAME,
        api_key=ANTHROPIC_API_KEY,
        max_tokens=MAX_TOKENS
    )

def get_llm(provider: str = "claude"):
    """
    Get LLM instance based on provider.
    
    Args:
        provider: "gemini", "openai", or "claude"
    
    Returns:
        LLM instance
    """
    llm_map = {
        'gemini': create_gemini_llm,
        'openai': create_openai_llm,
        'claude': create_claude_llm
    }
    
    llm_creator = llm_map.get(provider.lower())
    if llm_creator is None:
        raise ValueError(f"Unknown LLM provider: {provider}")
    
    return llm_creator()

def generate_answer(query: str, contexts: List[str], provider: str = "claude") -> Dict:
    """
    Generate answer using LLM with retrieved contexts.
    
    Args:
        query: User query
        contexts: Retrieved text chunks
        provider: LLM provider ("claude", "openai", "gemini")
    
    Returns:
        Dictionary with answer, contexts, and reasoning
    """
    # Get LLM
    llm = get_llm(provider)
    
    # Format contexts
    context_text = "\n\n".join([f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])
    
    # Get prompt template
    prompt = get_rag_prompt()
    
    # Generate response
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"query": query, "context_text": context_text})
    
    # Parse response
    answer = response.split("Reasoning:")[0].replace("Answer:", "").strip() if "Reasoning:" in response else response
    reasoning = response.split("Reasoning:")[1].strip() if "Reasoning:" in response else "Answer based on retrieved contexts."
    
    result = {
        "answer": answer,
        "contexts": [ctx[:200] + "..." for ctx in contexts],
        "reasoning": reasoning
    }
    
    return result