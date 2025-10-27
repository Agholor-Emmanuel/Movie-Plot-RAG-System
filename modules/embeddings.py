# embeddings.py
from typing import List
from langchain_community.vectorstores import FAISS
from utils.db_utils import create_vector_database

def create_embeddings(text_chunks: List[str], model: str = "HuggingFace") -> FAISS:
    """
    Create embeddings and store in FAISS vector database.
    
    Args:
        text_chunks: List of text chunks to embed
        model: Embedding model to use ("HuggingFace" or "OpenAI")
    
    Returns:
        FAISS vector store
    """
    vector_store = create_vector_database(text_chunks, model=model)
    return vector_store

if __name__ == "__main__":
    # Test embeddings
    from data_ingestion import ingest_data
    from data_preprocessing import preprocess_pipeline
    
    df = ingest_data(download=False)
    chunks = preprocess_pipeline(df)
    vector_store = create_embeddings(chunks)
    print("✓ Embeddings created and stored successfully")