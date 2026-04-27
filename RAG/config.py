
# LLM Backend Configuration
# Set to True for Ollama (free, local), False for OpenAI (paid, cloud)
USE_OLLAMA = True

# Model Settings
if USE_OLLAMA:
    # Ollama models
    LLM_MODEL = "llama3.2"           # For answer generation
    EMBEDDING_MODEL = "nomic-embed-text"  # For embeddings
    OLLAMA_HOST = "http://localhost:11434"
else:
    # OpenAI models
    LLM_MODEL = "gpt-4"              # For answer generation
    EMBEDDING_MODEL = "text-embedding-3-small"  # For embeddings
    # API key should be in environment variable: OPENAI_API_KEY

# Data Collection Settings
DEFAULT_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
DATA_FETCH_DAYS = 60  # Number of days of historical data to fetch

# Chunking Settings
DAILY_CHUNK_ENABLED = True
WEEKLY_CHUNK_ENABLED = True
CHUNK_OVERLAP = 0  # No overlap for financial data (discrete days)

# Vector Database Settings
VECTOR_DB_DIR = "vector_db"
FAISS_INDEX_TYPE = "IndexFlatL2"  # Exact search with L2 distance
# Alternative: "IndexIVFFlat" for faster approximate search

# Retrieval Settings
TOP_K_CHUNKS = 5  # Number of chunks to retrieve
SIMILARITY_THRESHOLD = None  # Optional: only return chunks above this similarity

# Generation Settings
TEMPERATURE = 0.3  # Lower = more deterministic, higher = more creative
MAX_TOKENS = 1000  # Maximum tokens in generated response

# Evaluation Settings
EVALUATION_OUTPUT_DIR = "evaluation_results"
SAVE_INTERMEDIATE_RESULTS = True

# Logging
VERBOSE = True  # Print progress messages
LOG_API_CALLS = False  # Log all LLM API calls (for debugging)

# Performance
BATCH_EMBEDDING_SIZE = 10  # Number of chunks to embed at once
CACHE_EMBEDDINGS = True  # Cache embeddings to avoid re-computation