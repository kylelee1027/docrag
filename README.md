# DocRAG — Local Document RAG Chatbot

A command-line tool that indexes documents on your computer, vectorizes them into a local database, and lets you ask a chatbot questions about their contents and locations.

## What It Does

1. **Crawl** — Recursively scan directories for supported file types
2. **Chunk** — Split documents into overlapping text chunks
3. **Embed** — Convert chunks into vector embeddings using a local model
4. **Store** — Persist vectors + metadata in a local vector database
5. **Retrieve** — Given a user query, find the most relevant chunks
6. **Generate** — Feed retrieved chunks to an LLM to produce a grounded answer with file paths

## Project Structure

```
docrag/
├── README.md
├── requirements.txt
├── config.py              # All tuneable constants in one place
├── main.py                # CLI entry point (argparse)
├── indexer/
│   ├── __init__.py
│   ├── crawler.py         # Walk directories, filter by extension, detect changes
│   ├── readers.py         # Extract text from each file type
│   └── chunker.py         # Split text into overlapping chunks
├── vectorstore/
│   ├── __init__.py
│   ├── embedder.py        # Wrapper around the embedding model
│   └── store.py           # ChromaDB (or FAISS) operations: add, query, delete
├── chat/
│   ├── __init__.py
│   ├── retriever.py       # Query the vector store + re-rank results
│   └── generator.py       # Build prompt, call LLM, format answer
└── tests/
    ├── test_chunker.py
    ├── test_readers.py
    └── test_retriever.py
```

## CLI Interface (target)

```bash
# Index a folder (recursively)
docrag index ~/Documents

# Index a single file
docrag index ~/notes/meeting.md

# Re-index (only changed files)
docrag index ~/Documents --incremental

# Interactive chat
docrag chat

# One-shot question
docrag ask "Where is my tax return from 2023?"

# Show what's indexed
docrag status

# Clear the database
docrag clear
```

---

## Build Plan — Suggested Order

Work through these phases in order. Each one is independently testable.

### Phase 1: File Reading (`indexer/readers.py`)

Build a function `read_file(path: str) -> str` that extracts plain text from each supported format. Start with `.txt` and `.md`, then add formats one at a time.

Supported formats to target:
- `.txt`, `.md` — read directly
- `.pdf` — use `pypdf` or `pdfplumber`
- `.docx` — use `python-docx`
- `.html` — use `beautifulsoup4` (strip tags)
- `.csv` — use stdlib `csv`, join rows into text
- `.py`, `.js`, `.ts`, `.json`, `.yaml` — read directly (treat as text)

Each reader should return a plain string. Wrap the dispatch logic in a dictionary mapping extensions to reader functions so it's easy to add new formats later.

### Phase 2: Chunking (`indexer/chunker.py`)

Build a function `chunk_text(text: str) -> list[dict]` that splits text into overlapping pieces.

Key decisions:
- **Chunk size**: ~500 characters is a good starting point. Too small = noisy retrieval. Too large = diluted relevance.
- **Overlap**: ~50–80 characters so you don't lose context at boundaries.
- **Strategy**: Start with simple character-based splitting on sentence boundaries. Upgrade to recursive text splitting later if needed (LangChain's `RecursiveCharacterTextSplitter` is a good reference for the algorithm, but implement it yourself).

Each chunk should carry metadata:
```python
{
    "text": "...",
    "source_path": "/absolute/path/to/file",
    "filename": "file.pdf",
    "chunk_index": 3,
    "file_hash": "sha256:abc...",   # for incremental re-indexing
    "indexed_at": "2025-05-01T..."
}
```

### Phase 3: Embedding (`vectorstore/embedder.py`)

Wrap a local embedding model so the rest of the code doesn't depend on the specific model.

```python
class Embedder:
    def __init__(self, model_name: str): ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...
```

Recommended model: **`all-MiniLM-L6-v2`** via `sentence-transformers`. It's small (~80 MB), fast on CPU, and produces 384-dim vectors. You can swap in a larger model later without changing anything else.

### Phase 4: Vector Store (`vectorstore/store.py`)

Use **ChromaDB** in persistent mode (stores to disk, no server needed). It handles vector storage, metadata filtering, and nearest-neighbor search.

Core operations to implement:
- `add_documents(chunks, embeddings, metadatas)` — upsert chunks
- `query(query_embedding, n_results) -> list[dict]` — top-k retrieval
- `delete_by_source(source_path)` — remove all chunks from a file
- `get_stats() -> dict` — count of documents, unique files, etc.

Use the file hash in metadata so you can detect when a file has changed and needs re-indexing.

**Alternative**: FAISS + a JSON sidecar for metadata. More manual but no dependency on ChromaDB.

### Phase 5: Retrieval (`chat/retriever.py`)

Build the retrieval pipeline:

1. Embed the user's query using the same embedder
2. Query the vector store for top-k results (start with k=5)
3. (Optional) Re-rank results — a simple approach is to use the embedding similarity score directly; a fancier approach is a cross-encoder re-ranker

Return a list of `(chunk_text, metadata, score)` tuples.

### Phase 6: LLM Generation (`chat/generator.py`)

Build the prompt and call the LLM.

System prompt template (customize to taste):
```
You are a helpful assistant that answers questions about the user's local
documents. You are given relevant excerpts along with their file paths.

Rules:
- Only answer based on the provided context.
- Always cite which file(s) your answer comes from, including the full path.
- If the context doesn't contain the answer, say so honestly.
```

User prompt template:
```
Context:
---
[Source: /path/to/file.pdf, chunk 3]
<chunk text here>
---
[Source: /path/to/notes.md, chunk 1]
<chunk text here>
---

Question: <user question>
```

Use the **Anthropic Python SDK** (`anthropic` package) to call Claude. You'll need an `ANTHROPIC_API_KEY` env variable. Alternatively, support OpenAI-compatible APIs or local models via `ollama` as a stretch goal.

### Phase 7: CLI (`main.py`)

Wire it all together with `argparse`. Each subcommand maps to a function:
- `index` → crawl + read + chunk + embed + store
- `chat` → loop of (input → retrieve → generate → print)
- `ask` → single-shot version of chat
- `status` → query store stats
- `clear` → drop the collection

### Phase 8 (Stretch): Improvements

- **Incremental indexing**: Hash each file, skip unchanged ones
- **File watching**: Use `watchdog` to auto-re-index on changes
- **Hybrid search**: Combine vector similarity with BM25 keyword search (ChromaDB supports `where` filters on metadata)
- **Conversation memory**: Keep a sliding window of chat history for follow-up questions
- **Source preview**: Open the file at the relevant location after answering
- **Web UI**: Simple Gradio or Streamlit frontend

---

## Key Design Decisions to Make

| Decision | Simple Option | Better Option |
|---|---|---|
| Vector DB | ChromaDB (zero config) | FAISS + custom metadata store |
| Embedding model | `all-MiniLM-L6-v2` (CPU, fast) | `nomic-embed-text` (better quality) |
| LLM | Claude via API | Local model via Ollama |
| Chunk strategy | Fixed-size character split | Recursive split on paragraphs → sentences → words |
| Re-ranking | None (use raw similarity) | Cross-encoder (`ms-marco-MiniLM`) |
| Change detection | File hash (SHA-256) | Hash + mtime for speed |

## Tips

- **Test retrieval before adding the LLM.** Build a `search` command that just returns chunks. If retrieval is bad, the LLM can't save you.
- **Print similarity scores** during development. If your top result has a score of 0.2 and #5 has 0.19, your embeddings aren't differentiating well — try larger chunks or a better model.
- **Watch your chunk boundaries.** A sentence cut in half across two chunks is useless. Split on sentence endings (`. `, `\n\n`) when possible.
- **Store absolute paths.** Relative paths break if you run the tool from different directories.
- **Normalize paths** with `os.path.abspath()` and `os.path.expanduser()` before storing.
