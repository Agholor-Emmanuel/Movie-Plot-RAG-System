# data_ingestion.py
import os
import pandas as pd
from config import KAGGLE_DATASET, DATA_PATH, SAMPLE_SIZE
import kaggle


def download_kaggle_data():
    """
    Download dataset from Kaggle and save to data folder.
    """
    try:
        # Authenticate and download
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(KAGGLE_DATASET, path=DATA_PATH, unzip=True)
        print(f"✓ Dataset downloaded to {DATA_PATH}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        raise

def load_movie_data(filepath: str = None, sample_size: int = SAMPLE_SIZE) -> pd.DataFrame:
    """
    Load the movie dataset and return a subset.
    
    Args:
        filepath: Path to the CSV file (default: data/wiki_movie_plots_deduped.csv)
        sample_size: Number of rows to sample
    
    Returns:
        DataFrame with Title and Plot columns
    """
    if filepath is None:
        filepath = os.path.join(DATA_PATH, 'wiki_movie_plots_deduped.csv')
    
    # Load data
    df = pd.read_csv(filepath)
    
    # Select relevant columns and remove nulls
    df = df[['Title', 'Plot']].dropna(subset=['Plot'])
    
    # Sample random rows
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    print(f"✓ Loaded {len(df)} movies")
    return df

def ingest_data(download: bool = False, sample_size: int = SAMPLE_SIZE) -> pd.DataFrame:
    """
    Complete data ingestion pipeline.
    
    Args:
        download: Whether to download from Kaggle first
        sample_size: Number of movies to sample
    
    Returns:
        DataFrame with movie data
    """
    if download:
        download_kaggle_data()
    
    return load_movie_data(sample_size=sample_size)

if __name__ == "__main__":
    # Test data ingestion
    df = ingest_data(download=True)
    print(df.head())