# utils/db_utils.py
from typing import List
#from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from config import EMBEDDING_MODEL, DEVICE, OPENAI_API_KEY

def get_embeddings(model: str = "HuggingFace"):
    """
    Get embeddings model based on provider.
    
    Args:
        model: "HuggingFace" or "OpenAI"
    
    Returns:
        Embeddings instance
    """
    model_to_embeddings = {
        'HuggingFace': HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': DEVICE}),
        'OpenAI': OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    }
    
    embeddings = model_to_embeddings.get(model)
    if embeddings is None:
        raise ValueError(f"Unknown embedding model: {model}")
    
    return embeddings

def create_vector_database(text_chunks: List[str], model: str = "HuggingFace") -> FAISS:
    """
    Create FAISS vector database from text chunks.
    
    Args:
        text_chunks: List of text chunks
        model: Embedding model ("HuggingFace" or "OpenAI")
    
    Returns:
        FAISS vector store
    """
    embeddings = get_embeddings(model)
    vector_database = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    
    print(f"✓ Created vector database with {len(text_chunks)} vectors")
    return vector_database