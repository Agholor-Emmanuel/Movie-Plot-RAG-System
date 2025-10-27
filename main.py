# main.py
import json
from typing import Dict
from langchain_community.vectorstores import FAISS
from modules.data_ingestion import ingest_data
from modules.data_preprocessing import preprocess_pipeline
from modules.embeddings import create_embeddings
from modules.retrieval import retrieve_and_generate
from config import TOP_K, DEFAULT_LLM_COMP

def rag_query(query: str, vector_store: FAISS, top_k: int = TOP_K, provider: str = DEFAULT_LLM_COMP) -> Dict:
    """
    Complete RAG pipeline: retrieve and generate answer.
    
    Args:
        query: User query
        vector_store: FAISS vector database
        top_k: Number of chunks to retrieve
        provider: LLM provider ('claude', 'openai', 'gemini')
    
    Returns:
        Structured JSON output with answer, contexts, and reasoning
    """
    result = retrieve_and_generate(query, vector_store, top_k, provider)
    return result

def main(queries: list[str]):
    """Main RAG pipeline execution."""
    print("Movie Plot RAG System")
    
    # Step 1: Data Ingestion
    print("\n[1/4] Data Ingestion...")
    df = ingest_data(download=False, sample_size=500)
    
    # Step 2: Data Preprocessing
    print("\n[2/4] Data Preprocessing...")
    chunks = preprocess_pipeline(df)
    
    # Step 3: Create Embeddings
    print("\n[3/4] Creating Embeddings...")
    vector_store = create_embeddings(chunks, model="HuggingFace")
    
    # Step 4: Query the RAG System
    print("\n[4/4] Querying RAG System...")
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        result = rag_query(query, vector_store, top_k=3, provider=DEFAULT_LLM_COMP)
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    queries = [
        "What movies feature artificial intelligence?",
        "Tell me about a movie with time travel",
        "Which movie has a detective solving crimes?"
    ]
    main(queries)