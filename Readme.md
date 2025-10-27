# 🎬 Movie Plot RAG System

A lightweight Retrieval-Augmented Generation (RAG) system that answers questions about movie plots using data from the Wikipedia Movie Plots dataset. The system retrieves relevant movie information and generates natural language answers using LLMs (Claude, OpenAI GPT, or Google Gemini).

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)
- [Example Output](#example-output)

## ✨ Features

- **Multi-LLM Support**: Switch between Claude, OpenAI GPT-4, or Google Gemini
- **Efficient Retrieval**: Uses FAISS for fast similarity search with OpenAI embeddings
- **Smart Chunking**: Automatically splits long movie plots while preserving context
- **Vector Store Caching**: Save and load embeddings for instant subsequent queries
- **Modular Design**: Clean, maintainable code structure
- **Rich Responses**: Returns structured JSON with answer, contexts, and reasoning
- **Logging**: Comprehensive logging to track system performance

## 🏗️ Architecture
```
Query → OpenAI Embedding → FAISS Retrieval → Context + Prompt → LLM → Structured Answer
```

1. **Data Ingestion**: Downloads and loads movie data from Kaggle
2. **Preprocessing**: Formats and chunks movie plots (1000 chars/chunk, 200 overlap)
3. **Embedding**: Converts text to vectors using OpenAI's embedding model
4. **Storage**: Stores vectors in FAISS for efficient retrieval (cached to disk)
5. **Retrieval**: Finds top-k relevant movie plots for queries
6. **Generation**: Uses LLM to generate answers with reasoning

## 📦 Prerequisites

- Python 3.8+
- Kaggle account (for dataset access)
- **OpenAI API key** (required for embeddings)
- API key for at least one LLM provider:
  - **Anthropic** (Claude) - Recommended
  - **OpenAI** (GPT-4)
  - **Google** (Gemini)

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd movies_plot_rag_agent
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

This will install:
- LangChain ecosystem (core, community, integrations)
- OpenAI Python SDK
- FAISS for vector storage
- Kaggle API client

## ⚙️ Configuration

### 1. Set Up Kaggle API

**Get your Kaggle API credentials:**
1. Go to [Kaggle.com](https://www.kaggle.com) and log in
2. Click your profile picture → **Settings**
3. Scroll to **API** section
4. Click **"Create New API Token"**
5. Download `kaggle.json`

### 2. Create Environment File

Create a `.env` file in the project root:
```env
# Kaggle Credentials
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key

# OpenAI API Key (REQUIRED for embeddings)
OPENAI_API_KEY=your_openai_api_key

# LLM API Keys (add at least one)
ANTHROPIC_API_KEY=your_claude_api_key
GEMINI_API_KEY=your_gemini_api_key
```

**Getting API Keys:**
- **OpenAI**: https://platform.openai.com/api-keys (Required!)
- **Claude**: https://console.anthropic.com/
- **Gemini**: https://aistudio.google.com/app/apikey

### 3. Configure Parameters (Optional)

Edit `config.py` to customize:
```python
# Data parameters
SAMPLE_SIZE = 500          # Number of movies to load
CHUNK_SIZE = 1000          # Characters per chunk
CHUNK_OVERLAP = 200        # Overlap between chunks

# Retrieval parameters
TOP_K = 8                  # Number of contexts to retrieve
DEFAULT_LLM_COMP = 'claude'  # Default LLM: 'claude', 'openai', or 'gemini'

# Paths
DATA_PATH = 'data/'
VECTOR_STORE_PATH = 'vector_store/'
```

## 🎯 Usage

### First Run (Build Vector Store)

On your first run, the system will build and cache the vector store:
```python
# In main.py
main(queries, load_from_cache=0)  # 0 = build from scratch
```
```bash
python main.py
```

This will:
1. Load 500 movies from Kaggle dataset
2. Chunk the movie plots
3. Create embeddings using OpenAI
4. Save to `vector_store/` directory (takes ~30-60 seconds)

### Subsequent Runs (Use Cache)

After the first run, use the cached vector store for instant queries:
```python
# In main.py
main(queries, load_from_cache=1)  # 1 = load from cache
```
```bash
python main.py
```


### Custom Queries

Edit `main.py` to add your own queries:
```python
if __name__ == "__main__":
    queries = [
        "What movies are about space exploration?",
        "Tell me about romantic comedies from the 1990s",
        "Which movies feature robots?"
    ]
    
    # Use cache
    main(queries, load_from_cache=1)
```

### Using Different LLM Providers

**Change default in `config.py`:**
```python
DEFAULT_LLM_COMP = 'claude'  # or 'openai' or 'gemini'
```

**Or in code:**
```python
result = rag_query(query, vector_store, provider="claude")
result = rag_query(query, vector_store, provider="openai")
result = rag_query(query, vector_store, provider="gemini")
```

### Rebuild Vector Store

If you change the data or want to rebuild:
```python
# In main.py
main(queries, load_from_cache=0)  # Force rebuild
```

## 📁 Project Structure
```
movies_plot_rag_agent/
├── config.py                 # Configuration and environment variables
├── main.py                   # Main execution script
├── requirements.txt          # Python dependencies
├── rag_system.log           # System logs (auto-generated)
├── .env                      # Environment variables (create this)
├── .gitignore               # Git ignore file
├── README.md                # This file
│
├── modules/                 # Core modules
│   ├── __init__.py
│   ├── data_ingestion.py    # Download and load data from Kaggle
│   ├── data_preprocessing.py # Format and chunk movie plots
│   ├── embeddings.py        # Create embeddings with caching
│   └── retrieval.py         # Retrieve contexts and generate answers
│
├── utils/                   # Utility functions
│   ├── __init__.py
│   ├── db_utils.py          # FAISS vector database operations
│   ├── llm_utils.py         # LLM initialization and response generation
│   └── prompt_utils.py      # Prompt templates
│
├── data/                    # Downloaded datasets (auto-created)
│   └── wiki_movie_plots_deduped.csv
│
└── vector_store/            # Cached embeddings (auto-created)
    ├── index.faiss
    └── index.pkl
```

## 🔄 How It Works

### 1. Data Ingestion
```python
from modules.data_ingestion import ingest_data
df = ingest_data(download=False, sample_size=500)
# Loads 500 random movies with Title and Plot columns
```

### 2. Preprocessing
```python
from modules.data_preprocessing import preprocess_pipeline
chunks = preprocess_pipeline(df)
# Output: ["Title: Movie A\nPlot: ...", "Title: Movie B\nPlot: ...", ...]
# Creates ~3.5 chunks per movie on average (1757 chunks from 500 movies)
```

### 3. Embedding & Storage
```python
from modules.embeddings import create_embeddings
vector_store = create_embeddings(chunks, model="OpenAI")
# Uses OpenAI's text-embedding-ada-002 model
# Automatically saves to vector_store/ directory
# Stores in FAISS for efficient similarity search
```

### 4. Loading from Cache
```python
from utils.db_utils import load_vector_store
vector_store = load_vector_store(embedding_model="OpenAI")
# Loads pre-computed embeddings from disk (instant!)
```

### 5. Query Processing
```python
from modules.retrieval import retrieve_and_generate
result = retrieve_and_generate(query, vector_store, top_k=8, provider="claude")
```

**Returns:**
```json
{
  "answer": "Natural language answer to the query",
  "contexts": ["Context 1...", "Context 2...", "Context 3..."],
  "reasoning": "Explanation of how the answer was formed"
}
```

### Modify Prompts

Edit `utils/prompt_utils.py` to customize how the LLM responds:
```python
def get_rag_prompt() -> ChatPromptTemplate:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Your custom system prompt here"),
        ("human", "Your custom user prompt template here")
    ])
    return prompt
```
### Clear Cache

To rebuild vector store from scratch:
```bash
# Delete cache
rm -rf vector_store/

# Or on Windows
rmdir /s vector_store
```

Then run with `load_from_cache=0`.


## 📄 License

This project is for educational purposes. The Wikipedia Movie Plots dataset is available under Kaggle's terms of use.

## 🙏 Acknowledgments

- Dataset: [Wikipedia Movie Plots](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots) by jrobischon
- Embeddings: OpenAI text-embedding-ada-002
- Vector Store: FAISS by Meta AI Research
- LLMs: Anthropic (Claude), OpenAI (GPT), Google (Gemini)

---

**Built with ❤️ using LangChain, OpenAI, and FAISS**