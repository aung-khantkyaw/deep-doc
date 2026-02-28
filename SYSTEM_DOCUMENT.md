# DeepDoc — System Document

> Version: 1.0.0 · Date: 2026-03-01  
> Local, privacy-first PDF study assistant with hybrid retrieval and role-based access.

---

## 1) Overview

DeepDoc lets users upload PDF files, then interact with content in two modes:

- 💬 Chat: question-answering over uploaded PDFs (RAG)
- 🎓 Quiz: question generation + answer evaluation

All LLM inference runs locally through Ollama.

---

## 2) Current Architecture (Implemented)

```text
Browser (Streamlit UI)
   │
   ▼
app.py
  ├─ Auth Gate (login/register)
  ├─ Sidebar Settings
  │   ├─ Admin: editable settings + Fast/Quality/Custom mode
  │   └─ User: read-only settings
  ├─ Upload & Process
  │   ├─ PDF save -> data/uploads/
  │   ├─ PDF parse + chunk -> engine/processor.py
  │   ├─ Hybrid index build -> engine/retriever.py
  │   └─ Process trace UI + terminal logs
  ├─ Chat Tab
  │   ├─ Hybrid retrieval (BM25 + vector)
  │   ├─ Retrieval trace UI (query/retrieve/fuse/context)
  │   └─ LLM answer -> engine/llm_chain.py
  └─ Quiz Tab
      ├─ Generate questions from retrieved context
      └─ Evaluate user answers

auth/database.py -> SQLite (users, history, settings)
auth/manager.py  -> register/login, bcrypt hashing, settings API
```

---

## 3) Tech Stack

| Layer             | Technology                                 | Notes                                            |
| ----------------- | ------------------------------------------ | ------------------------------------------------ |
| UI                | Streamlit                                  | Main web interface                               |
| LLM Runtime       | Ollama                                     | Local inference via HTTP                         |
| Chains            | LangChain + langchain-ollama               | Prompt + runnable pipelines                      |
| PDF parsing       | pypdf                                      | Page text extraction                             |
| Chunking          | langchain-text-splitters                   | RecursiveCharacterTextSplitter                   |
| Dense embeddings  | sentence-transformers (`all-MiniLM-L6-v2`) | Local embedding model                            |
| Embedding wrapper | langchain-huggingface                      | Replaces deprecated community embeddings wrapper |
| Vector store      | ChromaDB                                   | Persistent vector index                          |
| Sparse retrieval  | rank_bm25 via BM25Retriever                | Keyword matching                                 |
| Auth hashing      | bcrypt                                     | Direct bcrypt usage                              |
| DB                | SQLite                                     | Users, history, settings                         |

---

## 4) Project Structure

```text
DeepDoc/
├── app.py
├── README.md
├── SYSTEM_DOCUMENT.md
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
├── .env / .evn.example
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

---

## 5) Module Reference

### 5.1 `app.py`

Main Streamlit entrypoint.

Key responsibilities:

- Session bootstrap and auth gate
- Role-based sidebar settings
- Document processing pipeline orchestration
- Chat and quiz interaction loops
- UI trace panels (processing + retrieval)
- Terminal debug logging for processing and chat

Notable implemented behavior:

- **Performance Mode (admin):**
  - Fast: `llama3.2:3b`, `top_k=3`, `chunk_size=512`, `temperature=0.2`
  - Quality: `llama3.1:8b`, `top_k=5`, `chunk_size=768`, `temperature=0.3`
  - Custom: manual controls + save button
- **Windows-safe reprocessing:** new unique Chroma subfolder per processing run to avoid file-lock delete failures.

### 5.2 `engine/processor.py`

Pipeline:

1. `load_pdf()` -> one `Document` per page with metadata `{source, page}`
2. `split_documents()` -> chunked docs with overlap
3. `process_pdf()` -> full load + split

### 5.3 `engine/retriever.py`

Defines `HybridRetriever(BaseRetriever)` with weighted Reciprocal Rank Fusion (RRF):

- BM25 retrieval (`BM25Retriever`)
- Vector retrieval (`Chroma` retriever)
- Fusion with configurable weights (`bm25_weight`, `vector_weight`)

Also stores `last_debug_info` for UI/terminal trace:

- Original/normalized query
- BM25/vector/fused counts
- Top preview snippets with source/page metadata

`build_hybrid_retriever(...)` builds and returns `HybridRetriever`.

### 5.4 `engine/llm_chain.py`

Implemented chains:

- `build_chat_chain(retriever, model, temperature, base_url)`
- `build_chat_chain_with_context(model, temperature, base_url)`
- `build_quiz_chain(model, temperature, base_url)`
- `build_eval_chain(model, base_url)`

Chat prompt behavior:

- Use context as primary source
- Return fallback sentence only when context is empty/clearly unrelated

### 5.5 `utils/helpers.py`

- Save uploaded files
- Render chat messages
- Parse generated quiz questions

### 5.6 `auth/database.py`

SQLite CRUD and table bootstrap for:

- `users`
- `chat_history`
- `system_settings`

### 5.7 `auth/manager.py`

- Registration/login workflow
- Password hashing/verification with `bcrypt`
- Default admin creation
- Typed settings read/write helpers

---

## 6) Retrieval Strategy (Current)

DeepDoc uses hybrid sparse+dense retrieval:

1. BM25 for lexical matching
2. Vector search for semantic matching
3. Weighted RRF fusion for final ranking

$$
\text{score}(d) = \sum_{r \in \{\text{BM25}, \text{Vector}\}} \frac{w_r}{k_{rrf} + \text{rank}_r(d)}
$$

Current defaults in retriever:

- BM25 weight: `0.4`
- Vector weight: `0.6`
- RRF constant: `60`

---

## 7) Data Flow

### 7.1 Process Documents

1. Read active settings from DB
2. Create user-specific index root: `data/chroma_db/user_<id>/`
3. Create new unique run directory: `idx_<uuid>`
4. Save uploaded PDFs to `data/uploads/`
5. Parse and chunk all files
6. Build hybrid retriever with new index folder
7. Save process trace to session state
8. Best-effort cleanup of older index folders

### 7.2 Chat Question

1. User sends question
2. Hybrid retriever returns fused chunks
3. UI shows retrieval trace (query/retrieve/fuse/context stats)
4. `build_chat_chain_with_context()` generates answer from same retrieved context
5. Question + answer are persisted in `chat_history`

### 7.3 Quiz

1. Retrieve broad concept context
2. Generate numbered questions
3. User answers per question
4. Evaluation chain returns structured feedback
5. Quiz interactions are persisted in `chat_history` (mode=`quiz`)

---

## 8) Session State Keys

- `user`
- `auth_page`
- `chat_history`
- `retriever`
- `processed`
- `quiz_questions`
- `quiz_idx`
- `quiz_active`
- `quiz_context`
- `process_trace`

---

## 9) Environment Variables

| Variable             | Default                  | Purpose               |
| -------------------- | ------------------------ | --------------------- |
| `OLLAMA_BASE_URL`    | `http://localhost:11434` | Ollama API endpoint   |
| `UPLOAD_DIR`         | `data/uploads`           | Uploaded PDF storage  |
| `CHROMA_PERSIST_DIR` | `data/chroma_db`         | Chroma root directory |

System settings such as `model_name`, `top_k`, `temperature`, `chunk_size`, `chunk_overlap` are stored in SQLite (`system_settings`).

---

## 10) Supported Models (UI Options)

- `llama3.1:8b`
- `llama3.2:3b`
- `mistral:7b`
- `phi3:mini`
- `phi4:14b`
- `gemma2:9b`
- `qwen2.5:7b`
- `deepseek-r1:7b`

---

## 11) Security & Access

- Passwords hashed with `bcrypt`
- Role-based UI controls:
  - Admin can modify global settings
  - User can view settings only
- App seeds default admin on first boot if none exists

Default admin credentials:

- username: `admin`
- password: `admin123`

Change immediately after first login.

---

## 12) Operational Notes

- On Windows, Chroma files can remain locked by active handles; DeepDoc avoids hard-delete of in-use index by using a fresh run folder.
- Processing and chat actions print diagnostics to terminal for debugging.
- Retrieval and processing traces are also visible in UI expanders.

---

## 13) Known Gaps / Next Improvements

- Optional admin-only visibility toggle for debug traces
- Retrieval score-level diagnostics in UI
- Streaming token output in chat
- Password change UI
- User management panel for admin
