# DeepDoc - AI-Powered Document Intelligence — System Document

> Version: 1.0.0 · Date: 2026-03-01  
> Local, privacy-first PDF study assistant with hybrid retrieval, study-room timeline, and role-based access.

---

## 1) Overview

DeepDoc supports two study modes from uploaded PDFs:

- Chat: document-grounded Q&A (RAG)
- Quiz: auto-generated questions + answer evaluation

Everything runs locally with Ollama.

---

## 2) Current Architecture

```text
Browser (Streamlit)
   │
   ▼
app.py
  ├─ Auth Gate (login/register)
  ├─ Sidebar (theme-aware styling, room timeline, app actions)
  ├─ Upload Settings Panel (admin editable / user read-only)
  ├─ Study Rooms Sidebar Timeline
  │   ├─ select existing room -> restore room chat context
  │   └─ new chat -> upload new file set -> create new room
  ├─ Upload & Process
  │   ├─ save files
  │   ├─ parse + chunk
  │   ├─ build HybridRetriever (BM25 + Vector)
  │   └─ show Processing Trace
  ├─ Chat Tab
  │   ├─ retrieve (BM25+Vector+RRF)
  │   ├─ show Retrieval Trace
  │   └─ generate grounded answer
  └─ Quiz Tab
      ├─ generate quiz questions
      └─ evaluate answers

auth/database.py -> SQLite (users, rooms, room files, history, settings)
auth/manager.py  -> auth + bcrypt + typed settings
engine/retriever.py -> custom BM25 n-gram + vector + weighted RRF
```

---

## 3) Tech Stack

| Layer             | Technology                                 | Notes                           |
| ----------------- | ------------------------------------------ | ------------------------------- |
| UI                | Streamlit                                  | Main web UI                     |
| LLM Runtime       | Ollama                                     | Local inference via HTTP        |
| Chains            | LangChain + langchain-ollama               | Prompt/runnable chains          |
| PDF parsing       | pypdf                                      | Page extraction                 |
| Chunking          | langchain-text-splitters                   | Recursive splitter              |
| Dense embeddings  | sentence-transformers (`all-MiniLM-L6-v2`) | Local embedding model           |
| Embedding wrapper | langchain-huggingface                      | Current supported wrapper       |
| Vector store      | ChromaDB                                   | Persistent per-room index       |
| Sparse retrieval  | BM25Retriever                              | Custom n-gram preprocess        |
| Auth hashing      | bcrypt                                     | Direct bcrypt usage             |
| Database          | SQLite                                     | Users, rooms, history, settings |

---

## 4) Data Model (SQLite)

### 4.1 Core Tables

- `users`
- `system_settings`
- `chat_history` (includes `room_id` migration column)

### 4.2 Study Room Tables

- `study_rooms`
  - room metadata (`title`, `chroma_dir`, model/settings snapshot, timestamps)
- `room_files`
  - uploaded file timeline per room (`file_name`, `file_path`, `uploaded_at`)

This enables:

- selecting old rooms and continuing chat
- creating new rooms with new uploads
- preserving room-specific histories

---

## 5) Retrieval Strategy (Implemented)

DeepDoc uses hybrid sparse+dense retrieval:

1. BM25 lexical retrieval
2. Vector semantic retrieval
3. Weighted Reciprocal Rank Fusion (RRF)

$$
\text{score}(d)=\sum_{r \in \{\text{BM25},\text{Vector}\}}\frac{w_r}{k_{rrf}+\text{rank}_r(d)}
$$

Defaults:

- BM25 weight: `0.4`
- Vector weight: `0.6`
- RRF constant: `60`

### BM25 n-gram preprocessing

Implemented custom preprocess function for BM25:

- normalization + tokenization
- unigram + bigram + trigram feature generation

Example:

- `polytechnic university maubin`
- → `polytechnic`, `university`, `maubin`, `polytechnic_university`, `university_maubin`, `polytechnic_university_maubin`

---

## 6) App Behavior

### 6.1 Settings

Admin performance modes:

- Fast: `llama3.2:3b`, `top_k=3`, `chunk_size=512`, `temperature=0.2`
- Quality: `llama3.1:8b`, `top_k=5`, `chunk_size=768`, `temperature=0.3`
- Custom: manual values

User role sees read-only settings.

Performance mode control is rendered in the upload settings panel (not in the sidebar).

### 6.2 Study Rooms Timeline

Sidebar section `Chats`:

- shows rooms in time order (`updated_at`)
- each room shows title/time/file count
- click room to load that room and continue existing chat
- `New Chat` resets runtime state for a new room

### 6.3 Process Documents

On upload + process:

1. create unique room index path: `data/chroma_db/user_<id>/idx_<uuid>`
2. parse/chunk all uploaded files
3. build retriever
4. create `study_rooms` entry
5. create `room_files` entries
6. bind room as `current_room_id`
7. show processing trace

### 6.4 Chat / Quiz Persistence

Messages are saved with room context:

- `save_message(..., room_id=<current_room_id>)`
- room reload restores room-specific chat history
- quiz messages are also room-scoped

### 6.5 Chat Layout Behavior

- Chat bubbles are rendered in a grouped container.
- Chat input is always displayed below the bubble group.
- Before processing documents, chat input remains visible but disabled.

### 6.6 Upload Panel Behavior

- `Upload & Process PDF` expander is always expanded.
- Processing Trace remains in a separate collapsed expander.

### 6.7 Sidebar Theme Behavior

- Sidebar styles are theme-aware for light/dark mode.
- Text and button styles follow current Streamlit theme variables.

---

## 7) Debug/Trace UI

### 7.1 Processing Trace

Shows:

- uploaded files
- per-file chunk counts
- total chunks
- retrieval settings + index path

### 7.2 Retrieval Trace

Shows:

- original/normalized query
- BM25 n-gram token count + sample
- BM25/vector result counts + weights
- fused top chunk previews (source/page)
- context size sent to LLM

---

## 8) Session State Keys

- `user`
- `auth_page`
- `chat_history`
- `retriever`
- `current_room_id`
- `processed`
- `quiz_questions`
- `quiz_idx`
- `quiz_active`
- `quiz_context`
- `process_trace`

---

## 9) Environment Variables

| Variable             | Default                  | Purpose         |
| -------------------- | ------------------------ | --------------- |
| `OLLAMA_BASE_URL`    | `http://localhost:11434` | Ollama endpoint |
| `UPLOAD_DIR`         | `data/uploads`           | PDF storage     |
| `CHROMA_PERSIST_DIR` | `data/chroma_db`         | Chroma root     |

---

## 10) Supported Models (Current UI)

- `llama3.1:8b`
- `llama3.2:3b`
- `phi3:mini`

---

## 11) Operational Notes

- Windows file-lock safe approach: never hard-delete active index; use per-run unique folder.
- Terminal logs include processing and chat/retrieval diagnostics.
- Default admin created on first run (`admin` / `admin123`).

---

## 12) Next Improvements

- Admin-only toggle for debug trace visibility
- Room rename/delete from sidebar
- Streaming answer output
- Admin user management UI
