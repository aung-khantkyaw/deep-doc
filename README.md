# DeepDoc - AI-Powered Document Intelligence

DeepDoc is a local RAG study app for PDF files using Streamlit + Ollama.

## Features

- User registration/login with bcrypt
- Role-based access:
  - **admin**: manage global settings (Performance + Embedding)
  - **user**: study with read-only settings
- PDF upload and processing
- Sidebar supports light/dark theme-aware styling
- Hybrid retrieval via custom `HybridRetriever`:
  - BM25 (with custom n-gram preprocessing)
  - Chroma vector retrieval
  - weighted RRF fusion
- **Chat** from uploaded documents
- **Quiz** generation + evaluation
- **Study Rooms timeline** in sidebar:
  - reopen old room and continue existing chat
  - start new room by uploading new files
- Chat bubbles are rendered in one grouped area
- Chat input box is always shown below the bubble group (disabled until docs are processed)
- **Upload & Process PDF** panel is always expanded
- Upload settings in **Upload & Process PDF** are read-only (display current admin configuration)
- UI traces:
  - Processing Trace
  - BM25 + Vector Retrieval Trace
- All inference runs locally through Ollama

## Performance & Embedding Controls

Admin controls are in **Setting & Help → Performance Model Control**.

Performance modes:

- **Fast** (auto-applied)
  - model: `llama3.2:3b`
  - embedding: `all-MiniLM-L6-v2`
  - `top_k=3`, `chunk_size=512`, `temperature=0.2`
- **Quality** (auto-applied)
  - model: `llama3.1:8b`
  - embedding: `nomic-embed-text`
  - `top_k=5`, `chunk_size=768`, `temperature=0.3`
- **Custom**
  - manual LLM + embedding + retrieval settings

Embedding options:

- `all-MiniLM-L6-v2`
- `nomic-embed-text`

## Study Rooms (Timeline + Continuation)

Each document processing run creates a room:

- `study_rooms` record (settings snapshot + index path)
- `room_files` records (uploaded files with timestamps)
- room-scoped chat/quiz history (via `room_id`)

This allows:

- selecting a previous room and continuing chat
- creating a new room with new uploads without losing old rooms

## How to use Study Rooms

1. Upload PDF file(s) and click **Process Documents**.

- DeepDoc creates a new study room automatically.

2. Ask questions in **Chat**.

- Messages are saved to that room.

3. Open **Chats** in the sidebar.

- Click any previous room to continue its existing chat history,
- or click **New Chat** to start a fresh room.

## Retrieval Pipeline

1. Parse PDF pages (`pypdf`)
2. Split chunks (`RecursiveCharacterTextSplitter`)
3. Build BM25 + vector retrievers
4. Fuse rankings with weighted RRF
5. Send fused context to LLM

### BM25 n-gram preprocessing

BM25 uses custom `preprocess_func` with:

- normalization
- unigram + bigram + trigram features

This improves phrase-level matching compared with plain unigram tokenization.

## Windows File-Lock Safe Indexing

Per processing run, DeepDoc creates a unique index folder:

- `data/chroma_db/user_<id>/idx_<uuid>/`

This avoids `PermissionError [WinError 32]` when old index files are still locked.

## Project Structure

```text
DeepDoc/
├── app.py
├── README.md
├── SYSTEM_DOCUMENT.md
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
├── .env
├── .env.example
├── auth/
│   ├── database.py
│   └── manager.py
├── engine/
│   ├── processor.py
│   ├── retriever.py
│   └── llm_chain.py
├── utils/
│   └── helpers.py
└── data/
    ├── uploads/
    ├── chroma_db/
    └── deepdoc.db
```

## Requirements

- Python 3.10+
- Docker Desktop (or Docker Engine)
- Docker Compose

## Start Ollama (Docker)

```bash
docker compose up -d ollama
docker exec -it ollama ollama pull llama3.1:8b
# optional models
docker exec -it ollama ollama pull llama3.2:3b
docker exec -it ollama ollama pull phi3:mini
docker exec -it ollama ollama pull nomic-embed-text
```

## Local Setup

1. Create and activate virtual environment

```bash
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Windows (cmd)
.venv\Scripts\activate.bat
# Git Bash / Linux / macOS
source .venv/bin/activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Create environment file

```bash
copy .env.example .env
```

Set variables in `.env` (minimum: `OLLAMA_BASE_URL`).

### Environment Variables

| Variable                  | Default                                  | Purpose                                                 |
| ------------------------- | ---------------------------------------- | ------------------------------------------------------- |
| `OLLAMA_BASE_URL`         | `http://localhost:11434`                 | Ollama endpoint for chat + ollama embeddings            |
| `OLLAMA_MODEL`            | `llama3.1:8b`                            | Default LLM seed value for first-run settings           |
| `DEFAULT_EMBEDDING_MODEL` | `all-MiniLM-L6-v2`                       | Default embedding seed value for first-run settings     |
| `OLLAMA_EMBED_MODEL`      | `nomic-embed-text`                       | Ollama embedding model name when selected               |
| `HF_EMBED_MODEL`          | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding repo when selected                |
| `UPLOAD_DIR`              | `data/uploads`                           | Uploaded PDF storage path                               |
| `CHROMA_PERSIST_DIR`      | `data/chroma_db`                         | Chroma persistence root                                 |
| `TOP_K`                   | `4`                                      | Default top-k seed value for first-run settings         |
| `CHUNK_SIZE`              | `512`                                    | Default chunk size seed value for first-run settings    |
| `CHUNK_OVERLAP`           | `80`                                     | Default chunk overlap seed value for first-run settings |
| `TEMPERATURE`             | `0.2`                                    | Default temperature seed value for first-run settings   |

## Offline Mode Checklist

Use this checklist if you want DeepDoc to run without internet after initial setup.

1. Install dependencies once while online.
2. Pull required Ollama models once:

```bash
docker exec -it ollama ollama pull llama3.1:8b
docker exec -it ollama ollama pull nomic-embed-text
```

3. If you use `all-MiniLM-L6-v2`, run one processing session once while online to cache HuggingFace files.
4. Keep `.env` pointed to local Ollama (`OLLAMA_BASE_URL=http://localhost:11434`).
5. After that, run DeepDoc normally; chat/quiz/retrieval work offline with cached local models.

Notes:

- If HuggingFace DNS/download errors appear offline, switch embedding to `nomic-embed-text` in **Setting & Help**.
- First-time model download always requires internet.

## Run App

```bash
streamlit run app.py
```

Open `http://localhost:8501`.

## Docker (Optional: app + ollama)

```bash
docker compose up --build
```

Inside Docker network, app should use `OLLAMA_BASE_URL=http://ollama:11434`.

## Default Admin Account

Created automatically on first run:

```text
username: admin
password: admin123
```

Change this password after first login.

## Current Status

- Auth with bcrypt
- Role-based settings
- Hybrid BM25 + vector retrieval (custom `HybridRetriever`)
- BM25 n-gram preprocess (uni/bi/tri-gram)
- Study room timeline + room-based chat continuation
- Chat + quiz persistence by room
- Processing and retrieval trace panels
- Windows-safe per-run Chroma indexing
