# data_preprocessing.py
from typing import List
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from config import CHUNK_SIZE, CHUNK_OVERLAP

def preprocess_movies(df: pd.DataFrame) -> List[Document]:
    """
    Create Document objects with metadata to preserve movie titles.
    
    Args:
        df: DataFrame with Title and Plot columns
    
    Returns:
        List of Document objects with metadata
    """
    documents = []
    
    for idx, row in df.iterrows():
        # Create Document with plot as content and title as metadata
        doc = Document(
            page_content=row['Plot'],
            metadata={
                'title': row['Title'],
                'movie_id': idx
            }
        )
        documents.append(doc)
    
    print(f"✓ Preprocessed {len(documents)} movie documents")
    return documents

def create_text_chunks(documents: List[Document], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split documents into chunks while preserving title in each chunk.
    
    Args:
        documents: List of Document objects with metadata
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of text chunks with titles preserved
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]  # Added ". " to split at sentences
    )
    
    # Split documents while keeping metadata
    chunks = text_splitter.split_documents(documents)
    
    # Reformat each chunk with its title
    chunk_texts = []
    for chunk in chunks:
        # Each chunk gets the title from metadata
        formatted_chunk = f"Title: {chunk.metadata['title']}\nPlot: {chunk.page_content}"
        chunk_texts.append(formatted_chunk)
    
    print(f"✓ Created {len(chunk_texts)} chunks from {len(documents)} documents")
    
    # Show chunking statistics
    if len(chunk_texts) > len(documents):
        print(f"  → Some movies were split into multiple chunks")
        print(f"  → Average chunks per movie: {len(chunk_texts)/len(documents):.2f}")
    
    return chunk_texts

def preprocess_pipeline(df: pd.DataFrame) -> List[str]:
    """
    Complete preprocessing pipeline: format documents and chunk with metadata preservation.
    
    Args:
        df: DataFrame with movie data
    
    Returns:
        List of text chunks ready for embedding (each preserves movie title)
    """
    documents = preprocess_movies(df)
    chunks = create_text_chunks(documents)
    return chunks

if __name__ == "__main__":
    # Test preprocessing
    from data_ingestion import ingest_data
    
    df = ingest_data(download=False)
    chunks = preprocess_pipeline(df)
    
    print(f"\n{'='*60}")
    print("SAMPLE CHUNKS:")
    print('='*60)
    
    # Show first 3 chunks
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ---")
        print(chunk[:300] + "..." if len(chunk) > 300 else chunk)
    
    print(f"\n{'='*60}")
    print(f"Total chunks created: {len(chunks)}")