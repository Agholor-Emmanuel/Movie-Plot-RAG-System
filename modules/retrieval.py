# retrieval.py
from typing import List, Dict
from langchain_community.vectorstores import FAISS
from utils.llm_utils import generate_answer
from config import TOP_K, DEFAULT_LLM_COMP

def retrieve_relevant_chunks(query: str, vector_store: FAISS, top_k: int = TOP_K) -> List[str]:
    """
    Retrieve top-k relevant chunks for a query.
    
    Args:
        query: User query string
        vector_store: FAISS vector database
        top_k: Number of chunks to retrieve
    
    Returns:
        List of relevant text chunks
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.get_relevant_documents(query)
    
    contexts = [doc.page_content for doc in docs]
    print(f"✓ Retrieved {len(contexts)} chunks")
    return contexts

def retrieve_and_generate(query: str, vector_store: FAISS, top_k: int = TOP_K, provider: str = DEFAULT_LLM_COMP) -> Dict:
    """
    Retrieve relevant contexts and generate answer using LLM.
    
    Args:
        query: User query
        vector_store: FAISS vector database
        top_k: Number of chunks to retrieve
        provider: LLM provider ('claude', 'openai', 'gemini')
    
    Returns:
        Dictionary with answer, contexts, and reasoning
    """
    # Retrieve contexts
    contexts = retrieve_relevant_chunks(query, vector_store, top_k)
    
    # Generate answer
    result = generate_answer(query, contexts, provider)
    
    return result

if __name__ == "__main__":
    # Test retrieval
    from data_ingestion import ingest_data
    from data_preprocessing import preprocess_pipeline
    from embeddings import create_embeddings
    import json
    
    df = ingest_data(download=False)
    chunks = preprocess_pipeline(df)
    vector_store = create_embeddings(chunks)
    
    test_query = "What movies feature artificial intelligence?"
    result = retrieve_and_generate(test_query, vector_store)
    print(json.dumps(result, indent=2))