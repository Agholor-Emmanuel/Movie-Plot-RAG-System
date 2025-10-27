# main.py
import json
import time
import logging
from typing import Dict
from langchain_community.vectorstores import FAISS
from modules.data_ingestion import ingest_data
from modules.data_preprocessing import preprocess_pipeline
from modules.embeddings import create_embeddings
from modules.retrieval import retrieve_and_generate
from utils.db_utils import load_vector_store
from config import TOP_K, DEFAULT_LLM_COMP


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system.log'),  # Log to file
        logging.StreamHandler()  # Also print to console
    ]
)

logger = logging.getLogger(__name__)

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

def main(queries: list[str], load_from_cache: int = 1, embedding_model: str = "OpenAI"):
    """
    Main RAG pipeline execution.
    
    Args:
        queries: List of queries to ask
        load_from_cache: 1 = load from cache, 0 = rebuild from scratch
        embedding_model: "OpenAI" 
    """
    print("="*60)
    print("Movie Plot RAG System")
    print("="*60)
    
    logger.info(f"RAG system started")
    
    if load_from_cache == 1:
        # Load from cache
        print("\n📦 Loading vector store from cache...")
        vector_store = load_vector_store(embedding_model=embedding_model)
        
        if vector_store is None:
            print("⚠ Cache not found. Building from scratch...")
            load_from_cache = 0
    
    if load_from_cache == 0:
        # Build from scratch
        logger.info(f"🔨 Building vector store from scratch...")
        
        logger.info(f"[1/3] Data Ingestion...")
        df = ingest_data(download=False, sample_size=500)
        
        logger.info(f"[2/3] Data Preprocessing... ")
        chunks = preprocess_pipeline(df)
        
        logger.info(f"[3/3] Creating Embeddings..")
        vector_store = create_embeddings(chunks, model=embedding_model)
    
    # Query the RAG System
    print("\n" + "="*60)
    print("Querying RAG System")
    print("="*60)
    
    logger.info(f"Run Queries... ")
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        result = rag_query(query, vector_store, top_k=3, provider=DEFAULT_LLM_COMP)
        print(json.dumps(result, indent=2))
        logger.info(f"Queries answered")

if __name__ == "__main__":
    queries = [
        "What movies feature artificial intelligence?"
    ]
    
    # Use cache (fast!)
    main(queries, load_from_cache=1)
    
    #Rebuild from scratch
    #main(queries, load_from_cache=0)