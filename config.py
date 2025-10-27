# config.py
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Kaggle parameters
KAGGLE_USERNAME = os.environ.get('KAGGLE_USERNAME')
KAGGLE_KEY = os.environ.get('KAGGLE_KEY')
KAGGLE_DATASET = 'jrobischon/wikipedia-movie-plots'
DATA_PATH = 'data/'

# Data parameters
SAMPLE_SIZE = 500
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Embedding parameters
EMBEDDING_MODEL = 'sentence-transformers/all-mpnet-base-v2'
DEVICE = 'cpu'
VECTOR_STORE_PATH ='vector_store/' 

# LLM parameters
GEMINI_MODEL_NAME = 'models/gemini-2.0-flash-exp'
OPENAI_MODEL_NAME = 'gpt-4o'
CLAUDE_MODEL_NAME = 'claude-sonnet-4-20250514'
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
DEFAULT_LLM_COMP = 'claude'  # 'gemini', 'openai', or 'claude'

# Retrieval parameters
TOP_K = 8
MAX_TOKENS = 1024


