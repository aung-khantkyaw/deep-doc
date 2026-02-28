# DeepDoc

DeepDoc is a local RAG (Retrieval-Augmented Generation) study app for PDF files using Streamlit + Ollama.

## Features

- User registration and login (bcrypt password hashing)
- Role-based access:
  - **admin**: can change global system settings
  - **user**: can upload and study with read-only settings
- Upload one or more PDF files
- Hybrid retrieval with custom `HybridRetriever`:
  - BM25 keyword retrieval
  - Chroma vector retrieval
  - Weighted Reciprocal Rank Fusion (RRF)
- **💬 Chat**: ask questions from uploaded documents
- **🎓 Quiz**: generate questions + evaluate answers with structured feedback
- Per-user chat and quiz history stored in SQLite
- Debug visibility in UI:
  - Processing Trace
  - BM25 + Vector Retrieval Trace
- All inference runs locally via Ollama (no cloud LLM key required)

## Performance Modes

Admin sidebar supports 3 modes:

- **Fast** (auto-applied):
  - model: `llama3.2:3b`
  - `top_k=3`, `chunk_size=512`, `temperature=0.2`
- **Quality** (auto-applied):
  - model: `llama3.1:8b`
  - `top_k=5`, `chunk_size=768`, `temperature=0.3`
- **Custom**:
  - choose model and retrieval/generation settings manually

## Retrieval Pipeline

1. Parse PDF pages (`pypdf`)
2. Split into chunks (`RecursiveCharacterTextSplitter`)
3. Build hybrid retriever:
   - BM25Retriever
   - Chroma retriever with `all-MiniLM-L6-v2`
4. Fuse rankings via weighted RRF

## Windows File-Lock Safe Indexing

When processing documents multiple times, DeepDoc builds a fresh Chroma index per run:

- `data/chroma_db/user_<id>/idx_<uuid>/`

This avoids `PermissionError [WinError 32]` caused by deleting in-use index files.

## Project Structure

```text
DeepDoc/
├── .env
├── .evn.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── app.py
├── SYSTEM_DOCUMENT.md
├── README.md
├── auth/
│   ├── __init__.py
│   ├── database.py
│   └── manager.py
├── engine/
│   ├── __init__.py
│   ├── processor.py
│   ├── retriever.py
│   └── llm_chain.py
├── utils/
│   ├── __init__.py
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
docker exec -it ollama ollama pull llama3.2:3b
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
copy .evn.example .env
```

Set `OLLAMA_BASE_URL=http://localhost:11434` in `.env` when Streamlit runs on host.

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

- ✅ User auth with bcrypt
- ✅ Role-based settings with admin control
- ✅ Hybrid BM25 + vector retrieval (custom `HybridRetriever`)
- ✅ Chat + quiz workflows with per-user history
- ✅ Retrieval/process trace panels in UI
- ✅ Windows-safe per-run Chroma index strategy
