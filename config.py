"""
Central configuration — every tuneable constant in one place.
"""

import os
from pathlib import Path

# --- Paths ---
DB_DIR = os.path.expanduser("~/.local/share/docrag")
COLLECTION_NAME = "documents"

# --- Embedding ---
EMBED_MODEL = "all-MiniLM-L6-v2"   # 384-dim, ~80 MB, fast on CPU
EMBED_BATCH_SIZE = 64               # texts per batch when embedding

# --- Chunking ---
CHUNK_SIZE = 500          # target characters per chunk
CHUNK_OVERLAP = 60        # overlap between consecutive chunks
# Splitting priority: try paragraph breaks first, then sentences, then words
SPLIT_SEPARATORS = ["\n\n", "\n", ". ", " "]

# --- Retrieval ---
TOP_K = 5                 # number of chunks to retrieve per query
SIMILARITY_THRESHOLD = 0.25  # ignore chunks below this score (0-1, cosine)

# --- LLM ---
LLM_MODEL = "claude-sonnet-4-20250514"
LLM_MAX_TOKENS = 1024
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# --- Supported file types ---
SUPPORTED_EXTENSIONS = {
    # Plain text
    ".txt", ".md", ".rst", ".log",
    # Documents
    ".pdf", ".docx",
    # Web
    ".html", ".htm",
    # Data
    ".csv", ".json", ".yaml", ".yml",
    # Code (treat as text)
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".go", ".rs", ".c", ".cpp", ".h",
    ".sh", ".bash", ".zsh",
    ".sql", ".toml", ".ini", ".cfg",
}
