# utils/db_utils.py
import os
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from config import EMBEDDING_MODEL, DEVICE, OPENAI_API_KEY, VECTOR_STORE_PATH

def get_embeddings(model: str = "OpenAI"):
    """
    Get embeddings model based on provider.
    
    Args:
        model: "OpenAI" 
    
    Returns:
        Embeddings instance
    """
    model_to_embeddings = {
        'OpenAI': OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    }
    
    embeddings = model_to_embeddings.get(model)
    if embeddings is None:
        raise ValueError(f"Unknown embedding model: {model}")
    
    return embeddings

def create_vector_database(text_chunks: List[str], model: str = "OpenAI") -> FAISS:
    """
    Create FAISS vector database from text chunks.
    
    Args:
        text_chunks: List of text chunks
        model: Embedding model ("OpenAI")
    
    Returns:
        FAISS vector store
    """
    embeddings = get_embeddings(model)
    vector_database = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    
    print(f"✓ Created vector database with {len(text_chunks)} vectors")
    return vector_database

def save_vector_store(vector_store: FAISS, path: str = VECTOR_STORE_PATH):
    """
    Save FAISS vector store to disk.
    
    Args:
        vector_store: FAISS vector database
        path: Directory path to save the vector store
    """
    os.makedirs(path, exist_ok=True)
    vector_store.save_local(path)
    print(f"✓ Vector store saved to {path}")

def load_vector_store(path: str = VECTOR_STORE_PATH, embedding_model: str = "OpenAI") -> Optional[FAISS]:
    """
    Load FAISS vector store from disk.
    
    Args:
        path: Directory path to load the vector store from
        embedding_model: Embedding model to use (""OpenAI")
    
    Returns:
        FAISS vector store or None if not found
    """
    if not os.path.exists(path):
        print(f"⚠ No vector store found at {path}")
        return None
    
    try:
        embeddings = get_embeddings(embedding_model)
        vector_store = FAISS.load_local(
            path, 
            embeddings,
            allow_dangerous_deserialization=True  # Required for FAISS
        )
        print(f"✓ Vector store loaded from {path}")
        return vector_store
    except Exception as e:
        print(f"⚠ Error loading vector store: {e}")
        return None