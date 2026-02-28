# DeepDoc

DeepDoc is a local RAG (Retrieval-Augmented Generation) app for asking questions from PDF files using **Streamlit** + **Ollama**.

## Features

- Upload one or more PDF files
- Configure model and retrieval settings from the sidebar
- Ask questions from uploaded documents
- Show answer history in the UI
- Structured project layout for processor, retriever, and LLM chain logic

## Project Structure

```text
DeepDoc/
├── .env
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── app.py
├── engine/
│   ├── __init__.py
│   ├── processor.py
│   ├── retriever.py
│   └── llm_chain.py
├── utils/
│   ├── __init__.py
│   └── helpers.py
├── data/
│   ├── uploads/
│   └── chroma_db/
└── tests/
    └── test_retriever.py
```

## Requirements

- Python 3.10+
- Docker Desktop (or Docker Engine) running
- Docker Compose available

## Start Ollama (Docker)

Run Ollama as a container:

```bash
docker compose up -d ollama
```

Pull model inside the Ollama container (example):

```bash
docker exec -it ollama ollama pull llama3.1:8b
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

Set `OLLAMA_BASE_URL=http://localhost:11434` in `.env` when running Streamlit on host.

## Run App

```bash
streamlit run app.py
```

Open the URL shown in terminal (usually `http://localhost:8501`).

## Docker (Optional)

```bash
docker compose up --build
```

This runs both services (`ollama` + `deepdoc_app`) using `OLLAMA_BASE_URL=http://ollama:11434` inside Docker network.

## Current Status

- Streamlit UI scaffold is implemented in `app.py`.
- Backend wiring to `engine.processor`, `engine.retriever`, and `engine.llm_chain` is the next step.

## Next Development Steps

- Implement PDF loading + chunking in `engine/processor.py`
- Implement hybrid retrieval (BM25 + vector) in `engine/retriever.py`
- Implement Ollama response chain in `engine/llm_chain.py`
- Connect full RAG flow into `app.py`
