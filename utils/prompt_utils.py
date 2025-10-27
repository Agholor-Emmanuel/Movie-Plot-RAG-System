# utils/prompt_utils.py
from langchain.prompts import ChatPromptTemplate

def get_rag_prompt() -> ChatPromptTemplate:
    """
    Get the RAG prompt template for movie question answering.
    
    Returns:
        ChatPromptTemplate for RAG
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions about movies based on provided contexts.
You should provide accurate, informative answers based solely on the information given in the contexts."""),
        ("human", """Based on the following movie information, answer the question.

Question: {query}

Contexts:
{context_text}

Provide your response in the following format:
Answer: [Provide a clear, natural language answer to the question]
Reasoning: [Brief explanation of how you used the contexts to form your answer]""")
    ])
    
    return prompt

def get_system_prompt() -> str:
    """Get the system prompt for the LLM."""
    return """You are a helpful assistant that answers questions about movies based on provided contexts.
You should provide accurate, informative answers based solely on the information given in the contexts."""

def get_user_prompt_template() -> str:
    """Get the user prompt template."""
    return """Based on the following movie information, answer the question.

Question: {query}

Contexts:
{context_text}

Provide your response in the following format:
Answer: [Provide a clear, natural language answer to the question]
Reasoning: [Brief explanation of how you used the contexts to form your answer]"""