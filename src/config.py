import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TENDERS_DIR = os.path.join(BASE_DIR, "Tenders")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# Chunking configurations
DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 120

# Embedding model configurations
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Reranker configurations (Cross-Encoder)
USE_RERANKER = True
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"
CANDIDATE_POOL_SIZE = 25

# BM25 configurations
BM25_INDEX_PATH = os.path.join(BASE_DIR, "bm25_index.json")
STOP_WORDS = {"what", "is", "the", "of", "for", "this", "do", "we", "need", "in", "a", "an", "to", "how", "much", "are", "on", "at", "by"}

# Try to load environment variables from .env file if present at root
env_file = os.path.join(BASE_DIR, ".env")
if os.path.exists(env_file):
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip().strip('"').strip("'")

# LLM configurations (Groq)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
DEFAULT_LLM_MODEL = "llama-3.1-8b-instant"  # standard fast, free-tier model on Groq
