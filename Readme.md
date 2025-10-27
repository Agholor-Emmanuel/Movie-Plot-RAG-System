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
- **Efficient Retrieval**: Uses FAISS for fast similarity search
- **Smart Chunking**: Automatically splits long movie plots while preserving context
- **Modular Design**: Clean, maintainable code structure
- **Rich Responses**: Returns structured JSON with answer, contexts, and reasoning

## 🏗️ Architecture
```
Query → Embedding → FAISS Retrieval → Context + Prompt → LLM → Structured Answer
```

1. **Data Ingestion**: Downloads and loads movie data from Kaggle
2. **Preprocessing**: Formats and chunks movie plots (1000 chars/chunk, 200 overlap)
3. **Embedding**: Converts text to vectors using Sentence Transformers
4. **Storage**: Stores vectors in FAISS for efficient retrieval
5. **Retrieval**: Finds top-k relevant movie plots for queries
6. **Generation**: Uses LLM to generate answers with reasoning

## 📦 Prerequisites

- Python 3.8+
- Kaggle account (for dataset access)
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
- Sentence Transformers for embeddings
- FAISS for vector storage
- PyTorch (CPU version)
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

# LLM API Keys (add at least one)
ANTHROPIC_API_KEY=your_claude_api_key
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
```

**Getting API Keys:**
- **Claude**: https://console.anthropic.com/
- **OpenAI**: https://platform.openai.com/api-keys
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
DEFAULT_LLM_COMP = 'openai'  # Default LLM: 'claude', 'openai', or 'gemini'

# Embedding model
EMBEDDING_MODEL = 'sentence-transformers/all-mpnet-base-v2'
```

## 🎯 Usage

### Basic Usage

Run the system with default queries:
```bash
python main.py
```

### Custom Queries

Edit `main.py` to add your own queries:
```python
if __name__ == "__main__":
    queries = [
        "What movies are about space exploration?",
        "Tell me about romantic comedies",
        "Which movies feature robots?"
    ]
    main(queries)
```

### Using Different LLM Providers

**In code:**
```python
# Use Claude 
result = rag_query(query, vector_store, provider="claude")

# Use OpenAI
result = rag_query(query, vector_store, provider="openai")

# Use Gemini
result = rag_query(query, vector_store, provider="gemini")
```

**Or change default in `config.py`:**
```python
DEFAULT_LLM_COMP = 'gemini'  # Change to 'openai' or 'claude'
```

### First-Time Setup (Download Dataset)

To download the dataset from Kaggle on first run:
```python
# In main.py, change:
df = ingest_data(download=True, sample_size=500)  # Set download=True
```

After first download, change back to `download=False` to use cached data.

## 📁 Project Structure
```
movies_plot_rag_agent/
├── config.py                 # Configuration and environment variables
├── main.py                   # Main execution script
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (create this)
├── .gitignore               # Git ignore file
├── README.md                # This file
│
├── modules/                 # Core modules
│   ├── __init__.py
│   ├── data_ingestion.py    # Download and load data from Kaggle
│   ├── data_preprocessing.py # Format and chunk movie plots
│   ├── embeddings.py        # Create embeddings
│   └── retrieval.py         # Retrieve contexts and generate answers
│
├── utils/                   # Utility functions
│   ├── __init__.py
│   ├── db_utils.py          # FAISS vector database operations
│   ├── llm_utils.py         # LLM initialization and response generation
│   └── prompt_utils.py      # Prompt templates
│
└── data/                    # Downloaded datasets (auto-created)
    └── wiki_movie_plots_deduped.csv
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
vector_store = create_embeddings(chunks, model="HuggingFace")
# Uses sentence-transformers to create embeddings
# Stores in FAISS for efficient similarity search
```

### 4. Query Processing
```python
from modules.retrieval import retrieve_and_generate
result = retrieve_and_generate(query, vector_store, top_k=3, provider="claude")
```

**Returns:**
```json
{
  "answer": "Natural language answer to the query",
  "contexts": ["Context 1...", "Context 2...", "Context 3..."],
  "reasoning": "Explanation of how the answer was formed"
}
```


### Adjust Chunk Size

For longer contexts:
```python
CHUNK_SIZE = 1500        # Increase for more context
CHUNK_OVERLAP = 300      # Increase overlap proportionally
```

### Retrieve More Contexts
```python
TOP_K = 10  # Retrieve top 10 instead of top 8
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


## 📊 Example Output
```
============================================================
Query: What movies feature artificial intelligence?
============================================================
✓ Retrieved 3 chunks
{
  "answer": "Based on the provided contexts, 'Ex Machina' features artificial intelligence as a central theme. The movie involves a programmer who is invited to administer the Turing test to an intelligent humanoid robot.",
  "contexts": [
    "Title: Ex Machina\nPlot: A young programmer is selected to participate in a breakthrough experiment in synthetic intelligence by evaluating the human qualities of a highly advanced humanoid A.I....",
    "Title: 2001: A Space Odyssey\nPlot: Humanity finds a mysterious, obviously artificial, artifact buried beneath the Lunar surface and, with the intelligent computer HAL 9000...",
    "Title: Blade Runner\nPlot: In the futuristic year of 2019, Los Angeles has become a dark and depressing metropolis, filled with urban decay. Rick Deckard, an ex-cop, is a 'Blade Runner'..."
  ],
  "reasoning": "I searched through the retrieved contexts and found that 'Ex Machina' explicitly deals with artificial intelligence, featuring an AI robot undergoing the Turing test. Other contexts like '2001: A Space Odyssey' with HAL 9000 and 'Blade Runner' with replicants also involve AI themes."
}
```

## 📝 Notes

- **Dataset**: 34,886 movies from Wikipedia (using 500 sample by default)
- **Embedding Dimension**: 768 (all-mpnet-base-v2) or 384 (all-MiniLM-L6-v2)
- **Average Response Time**: 2-5 seconds per query
- **Token Usage**: ~500-1000 tokens per query (varies by LLM)


## 🙏 Acknowledgments

- Dataset: [Wikipedia Movie Plots](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots) by jrobischon
- Embeddings: Sentence Transformers by UKPLab
- Vector Store: FAISS by Meta AI Research
- LLMs: Anthropic (Claude), OpenAI (GPT), Google (Gemini)

---

**Built with ❤️ using LangChain, FAISS, and Sentence Transformers**