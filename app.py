"""DeepDoc - AI-Powered Document Intelligence — NotebookLM-style PDF study assistant with user auth.

Roles:
  admin — can change system settings (model, retrieval params) + all user features
  user  — can upload PDFs, chat, take quizzes; settings are read-only
"""
from __future__ import annotations

import gc
import os
import shutil
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

from auth.database import (
    add_room_file,
    create_study_room,
    init_db,
    clear_user_history,
    get_room_files,
    get_study_room,
    get_user_history,
    list_user_study_rooms,
    save_message,
    touch_study_room,
)
from auth.manager import (
    change_password,
    register,
    login,
    create_default_admin,
    get_system_settings,
    save_system_settings,
    update_profile,
)
from engine.processor import process_pdf
from engine.retriever import build_hybrid_retriever
from engine.llm_chain import (
    build_chat_chain,
    build_chat_chain_with_context,
    build_quiz_chain,
    build_eval_chain,
)
from utils.helpers import save_uploaded_file, display_chat_message, parse_quiz_questions, get_pdf_preview

load_dotenv()

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
UPLOAD_DIR: str      = os.getenv("UPLOAD_DIR", "data/uploads")
CHROMA_DIR: str      = os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")
DEFAULT_EMBEDDING_MODEL: str = os.getenv("DEFAULT_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

MODEL_OPTIONS = [
    "llama3.1:8b",
    "llama3.2:3b",
    "phi3:mini",
]

EMBEDDING_OPTIONS = [
    "all-MiniLM-L6-v2",
    "nomic-embed-text",
]

FILE_AI_ICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="26" height="26" viewBox="0 0 24 24" '
    'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
    'stroke-linejoin="round" class="icon icon-tabler icons-tabler-outline icon-tabler-file-ai">'
    '<path stroke="none" d="M0 0h24v24H0z" fill="none"/>'
    '<path d="M14 3v4a1 1 0 0 0 1 1h4" />'
    '<path d="M10 21h-3a2 2 0 0 1 -2 -2v-14a2 2 0 0 1 2 -2h7l5 5v4" />'
    '<path d="M14 21v-4a2 2 0 1 1 4 0v4" />'
    '<path d="M14 19h4" />'
    '<path d="M21 15v6" />'
    '</svg>'
)


def inject_custom_styles() -> None:
    """Apply custom card-style UI for sidebar and key blocks."""
    st.markdown(
        """
        <style>
        :root {
            --dd-sidebar-bg-start: var(--secondary-background-color);
            --dd-sidebar-bg-end: var(--secondary-background-color);
            --dd-sidebar-border: rgba(148, 163, 184, 0.24);
            --dd-sidebar-text: var(--text-color);
            --dd-btn-bg: rgba(148, 163, 184, 0.16);
            --dd-btn-text: var(--text-color);
            --dd-btn-hover-bg: rgba(148, 163, 184, 0.28);
            --dd-tertiary-text: var(--text-color);
            --dd-tertiary-hover-text: var(--text-color);
            --dd-input-bg: rgba(15, 23, 42, 0.75);
            --dd-input-border: rgba(148, 163, 184, 0.25);
            --dd-expander-bg: rgba(15, 23, 42, 0.22);
            --dd-expander-border: rgba(148, 163, 184, 0.28);
        }

        [data-theme="light"] {
            --dd-sidebar-text: #111827;
            --dd-btn-text: #111827;
            --dd-tertiary-text: #111827;
            --dd-tertiary-hover-text: #111827;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, var(--dd-sidebar-bg-start) 0%, var(--dd-sidebar-bg-end) 100%) !important;
            border-right: 1px solid var(--dd-sidebar-border);
        }

        [data-testid="stSidebar"] > div:first-child {
            background: linear-gradient(180deg, var(--dd-sidebar-bg-start) 0%, var(--dd-sidebar-bg-end) 100%) !important;
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span {
            color: var(--dd-sidebar-text);
        }

        [data-testid="stSidebar"] [data-testid="stButton"] > button {
            border-radius: 12px;
            border: none;
            margin: 2px 0;
            background: var(--dd-btn-bg);
            color: var(--dd-btn-text);
            transition: all 0.15s ease;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        [data-testid="stSidebar"] [data-testid="stButton"] > button:hover {
            border: none;
            background: var(--dd-btn-hover-bg);
        }

        [data-testid="stSidebar"] [data-testid="stButton"] > button[kind="tertiary"] {
            border: none;
            background: transparent;
            color: var(--dd-tertiary-text);
        }

        [data-testid="stSidebar"] [data-testid="stButton"] > button[kind="tertiary"]:hover {
            border: none;
            background: transparent;
            color: var(--dd-tertiary-hover-text);
        }

        [data-testid="stSidebar"] [data-testid="stButton"] > button[kind="secondary"] {
            border: none;
            background: var(--dd-btn-bg);
            color: var(--dd-btn-text);
        }

        [data-testid="stSidebar"] [data-testid="stButton"] > button[kind="primary"] {
            border: 1px solid rgba(239, 68, 68, 0.8);
            color: var(--dd-btn-text);
            background: rgba(127, 29, 29, 0.2);
        }

        [data-testid="stSidebar"] [data-testid="stButton"] > button[kind="primary"]:hover {
            border: 1px solid rgba(248, 113, 113, 0.95);
            color: var(--dd-btn-text);
            background: rgba(153, 27, 27, 0.3);
        }

        [data-theme="light"] [data-testid="stSidebar"] [data-testid="stButton"] > button[kind="primary"] {
            border: 1px solid rgba(217, 119, 6, 0.9);
            color: var(--dd-btn-text);
            background: rgba(255, 237, 213, 0.95);
        }

        [data-theme="light"] [data-testid="stSidebar"] [data-testid="stButton"] > button[kind="primary"]:hover {
            border: 1px solid rgba(180, 83, 9, 0.95);
            color: var(--dd-btn-text);
            background: rgba(254, 215, 170, 0.95);
        }

        [data-testid="stSidebar"] [data-testid="stSelectbox"],
        [data-testid="stSidebar"] [data-testid="stSlider"],
        [data-testid="stSidebar"] [data-testid="stRadio"] {
            padding: 10px 12px;
            border-radius: 12px;
            border: 1px solid var(--dd-input-border);
            background: var(--dd-input-bg);
            margin-bottom: 8px;
        }

        [data-testid="stSidebar"] hr {
            border-color: var(--dd-input-border);
        }

        [data-testid="stExpander"] {
            border: 1px solid var(--dd-expander-border);
            border-radius: 14px;
            background: var(--dd-expander-bg);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Bootstrap DB once per process
# ---------------------------------------------------------------------------

init_db()
create_default_admin()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def initialize_state() -> None:
    defaults: dict = {
        # Auth
        "user": None,             # dict: {id, username, email, role} | None
        "auth_page": "login",     # 'login' | 'register'
        "active_page": "chat",    # 'chat' | 'settings'
        # RAG
        "chat_history": [],
        "retriever": None,
        "current_room_id": None,
        "processed": False,
        "quiz_questions": [],
        "quiz_idx": 0,
        "quiz_active": False,
        "quiz_context": "",
        "process_trace": None,
        "uploader_nonce": 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def is_logged_in() -> bool:
    return st.session_state.user is not None


def is_admin() -> bool:
    return is_logged_in() and st.session_state.user["role"] == "admin"


def _reset_runtime_state(keep_room: bool = True) -> None:
    old_retriever = st.session_state.get("retriever")
    st.session_state.retriever = None
    if old_retriever is not None:
        del old_retriever
    gc.collect()

    st.session_state.chat_history = []
    st.session_state.quiz_questions = []
    st.session_state.quiz_idx = 0
    st.session_state.quiz_active = False
    st.session_state.quiz_context = ""
    st.session_state.processed = False
    st.session_state.process_trace = None
    if not keep_room:
        st.session_state.current_room_id = None


def load_study_room(room_id: int) -> tuple[bool, str]:
    user_id = st.session_state.user["id"]
    room = get_study_room(room_id, user_id)
    if room is None:
        return False, "Study room not found."

    file_rows = get_room_files(room_id)
    if not file_rows:
        return False, "No files found in this study room."

    _reset_runtime_state(keep_room=True)

    all_chunks = []
    per_file_stats: list[dict] = []
    for rf in file_rows:
        file_path = rf["file_path"]
        if not os.path.exists(file_path):
            continue
        chunks = process_pdf(
            file_path,
            chunk_size=int(room["chunk_size"]),
            chunk_overlap=int(room["chunk_overlap"]),
        )
        all_chunks.extend(chunks)
        per_file_stats.append(
            {
                "file_name": rf["file_name"],
                "saved_path": file_path,
                "chunk_count": len(chunks),
            }
        )

    if not all_chunks:
        return False, "Could not load chunks from the files in this room."

    chroma_dir = room["chroma_dir"]
    embedding_model = (
        room["embedding_model"]
        if "embedding_model" in room.keys()
        else get_system_settings().get("embedding_model", DEFAULT_EMBEDDING_MODEL)
    )
    os.makedirs(chroma_dir, exist_ok=True)
    st.session_state.retriever = build_hybrid_retriever(
        all_chunks,
        persist_dir=chroma_dir,
        top_k=int(room["top_k"]),
        embedding_model=embedding_model,
    )
    st.session_state.current_room_id = room_id
    st.session_state.processed = True
    st.session_state.chat_history = [
        {"role": r["role"], "content": r["content"]}
        for r in get_user_history(user_id, mode="chat", room_id=room_id)
    ]
    st.session_state.process_trace = {
        "uploaded_files": [r["file_name"] for r in file_rows],
        "per_file_stats": per_file_stats,
        "total_files": len(file_rows),
        "total_chunks": len(all_chunks),
        "model_name": room["model_name"],
        "embedding_model": embedding_model,
        "top_k": int(room["top_k"]),
        "chunk_size": int(room["chunk_size"]),
        "chunk_overlap": int(room["chunk_overlap"]),
        "bm25_weight": 0.4,
        "vector_weight": 0.6,
        "chroma_dir": chroma_dir,
    }
    return True, f"Loaded room: {room['title']}"


def render_room_timeline() -> None:
    user_id = st.session_state.user["id"]
    rooms = list_user_study_rooms(user_id)

    if not rooms:
        st.sidebar.caption("No chats yet. Upload a PDF to create one.")
        return

    current_room_id = st.session_state.get("current_room_id")
    for room in rooms:
        room_id = int(room["id"])
        is_current = room_id == current_room_id
        updated_at = room["updated_at"]
        file_count = int(room["file_count"] or 0)
        label = f"{room['title']} · {updated_at} · {file_count} file(s)"
        if st.sidebar.button(
            label,
            key=f"room_select_{room_id}",
            type="secondary" if is_current else "tertiary",
            use_container_width=True,
        ):
            ok, msg = load_study_room(room_id)
            if ok:
                st.sidebar.success(msg)
                st.session_state.active_page = "chat"
            else:
                st.sidebar.error(msg)
            st.rerun()


# ---------------------------------------------------------------------------
# Auth pages
# ---------------------------------------------------------------------------

def render_login_page() -> None:
    col = st.columns([1, 2, 1])[1]
    with col:
        st.markdown("## DeepDoc - AI-Powered Document Intelligence")
        st.caption("Sign in to continue")
        st.markdown("---")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login", type="primary", use_container_width=True):
            ok, msg, user = login(username, password)
            if ok:
                st.session_state.user = user
                st.session_state.active_page = "chat"
                st.session_state.current_room_id = None
                recent_rooms = list_user_study_rooms(user["id"], limit=1)
                if recent_rooms:
                    load_study_room(int(recent_rooms[0]["id"]))
                else:
                    _reset_runtime_state(keep_room=False)
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

        st.markdown("---")
        st.caption("Don't have an account?")
        if st.button("Register", use_container_width=True):
            st.session_state.auth_page = "register"
            st.rerun()


def render_register_page() -> None:
    col = st.columns([1, 2, 1])[1]
    with col:
        st.markdown("## DeepDoc - AI-Powered Document Intelligence — Register")
        st.markdown("---")

        username = st.text_input("Username")
        email    = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm  = st.text_input("Confirm Password", type="password")

        if st.button("Create Account", type="primary", use_container_width=True):
            if password != confirm:
                st.error("Passwords do not match.")
            else:
                ok, msg = register(username, email, password)
                if ok:
                    st.success(msg + " Please log in.")
                    st.session_state.auth_page = "login"
                    st.rerun()
                else:
                    st.error(msg)

        st.markdown("---")
        if st.button("Back to Login", use_container_width=True):
            st.session_state.auth_page = "login"
            st.rerun()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    user = st.session_state.user

    st.sidebar.markdown(
        (
            '<div style="display:flex;align-items:center;gap:8px;color:var(--text-color);margin-bottom:50px;">'
            f"{FILE_AI_ICON_SVG}"
            '<span style="font-size:2rem;font-weight:700;line-height:1;">DeepDoc</span>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    col1, col2 = st.sidebar.columns([1, 1])
    with col1:
        if st.button("➕ New Chat", use_container_width=True, help="Ctrl+K"):
            _reset_runtime_state(keep_room=False)
            st.session_state.uploader_nonce += 1
            st.session_state.active_page = "chat"
            st.rerun()
    with col2:
        if st.button("🗑️ Clear Data", use_container_width=True, type="secondary", help="Clear all user data"):
            user_id = st.session_state.user["id"]
            clear_user_history(user_id, mode="chat")
            clear_user_history(user_id, mode="quiz")
            _reset_runtime_state(keep_room=False)
            st.sidebar.success("All data cleared!")
            st.rerun()

    st.sidebar.markdown("### Chats")
    render_room_timeline()

    st.sidebar.markdown("---")
    if st.sidebar.button("Setting & Help", use_container_width=True):
        st.session_state.active_page = "settings"
        st.rerun()

    if st.sidebar.button("Logout", use_container_width=True, type="primary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


def render_upload_settings_panel() -> tuple[str, str, int, float, int]:
    settings = get_system_settings()
    model_name = settings["model_name"]
    embedding_model = settings.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
    top_k = settings["top_k"]
    temperature = settings["temperature"]
    chunk_size = settings["chunk_size"]
    st.markdown("###### Performance Mode")
    st.caption(
        f"LLM `{model_name}` · Embed `{embedding_model}` · TopK `{top_k}` · "
        f"Temperature `{temperature}` · Chunk `{chunk_size}`"
    )
    st.caption("Read-only here. Admin can change these in Setting & Help.")
    return model_name, embedding_model, top_k, temperature, chunk_size


def render_settings_page() -> None:
    user = st.session_state.user
    st.title("Setting & Help")
    st.caption("Manage your account profile and password")

    with st.form("profile_form"):
        st.markdown("#### Profile")
        new_username = st.text_input("Username", value=user["username"])
        new_email = st.text_input("Email", value=user["email"])
        save_profile = st.form_submit_button("Save Profile", type="primary")

    if save_profile:
        ok, msg, updated_user = update_profile(user["id"], new_username, new_email)
        if ok and updated_user is not None:
            st.session_state.user = updated_user
            st.success(msg)
        else:
            st.error(msg)

    with st.form("password_form"):
        st.markdown("#### Change Password")
        current_pw = st.text_input("Current Password", type="password")
        new_pw = st.text_input("New Password", type="password")
        confirm_pw = st.text_input("Confirm New Password", type="password")
        save_password = st.form_submit_button("Update Password")

    if save_password:
        if new_pw != confirm_pw:
            st.error("New password and confirm password do not match.")
        else:
            ok, msg = change_password(user["id"], current_pw, new_pw)
            if ok:
                st.success(msg)
            else:
                st.error(msg)

    st.markdown("#### Performance Model Control")
    if is_admin():
        settings = get_system_settings()
        fast_preset = {
            "model_name": "llama3.2:3b",
            "embedding_model": "all-MiniLM-L6-v2",
            "top_k": 3,
            "temperature": 0.2,
            "chunk_size": 512,
        }
        quality_preset = {
            "model_name": "llama3.1:8b",
            "embedding_model": "nomic-embed-text",
            "top_k": 5,
            "temperature": 0.3,
            "chunk_size": 768,
        }

        if (
            settings["model_name"] == fast_preset["model_name"]
            and settings.get("embedding_model") == fast_preset["embedding_model"]
            and settings["top_k"] == fast_preset["top_k"]
            and settings["temperature"] == fast_preset["temperature"]
            and settings["chunk_size"] == fast_preset["chunk_size"]
        ):
            default_mode = "Fast"
        elif (
            settings["model_name"] == quality_preset["model_name"]
            and settings.get("embedding_model") == quality_preset["embedding_model"]
            and settings["top_k"] == quality_preset["top_k"]
            and settings["temperature"] == quality_preset["temperature"]
            and settings["chunk_size"] == quality_preset["chunk_size"]
        ):
            default_mode = "Quality"
        else:
            default_mode = "Custom"

        perf_key = "settings_performance_mode"
        if perf_key not in st.session_state or st.session_state[perf_key] not in {"Fast", "Quality", "Custom"}:
            st.session_state[perf_key] = default_mode

        performance_mode = st.radio(
            "Mode",
            options=["Fast", "Quality", "Custom"],
            key=perf_key,
            horizontal=True,
        )

        chunk_overlap = settings["chunk_overlap"]
        if performance_mode == "Fast":
            st.caption(
                f"LLM `{fast_preset['model_name']}` · Embed `{fast_preset['embedding_model']}` "
            )
            st.caption(
                f"TopK `{fast_preset['top_k']}` · Temp `{fast_preset['temperature']}` · Chunk `{fast_preset['chunk_size']}`"
            )
            if st.button("Apply Fast", use_container_width=False):
                save_system_settings(
                    fast_preset["model_name"],
                    fast_preset["embedding_model"],
                    fast_preset["top_k"],
                    fast_preset["temperature"],
                    fast_preset["chunk_size"],
                    chunk_overlap,
                )
                st.success("Fast profile applied")
                st.rerun()
            
        elif performance_mode == "Quality":
            st.caption(
                f"LLM `{quality_preset['model_name']}` · Embed `{quality_preset['embedding_model']}` "
            )
            st.caption(
                f"TopK `{quality_preset['top_k']}` · Temp `{quality_preset['temperature']}` · Chunk `{quality_preset['chunk_size']}`"
            )
            if st.button("Apply Quality", use_container_width=False):
                save_system_settings(
                    quality_preset["model_name"],
                    quality_preset["embedding_model"],
                    quality_preset["top_k"],
                    quality_preset["temperature"],
                    quality_preset["chunk_size"],
                    chunk_overlap,
                )
                st.success("Quality profile applied")
                st.rerun()
            
        else:
            llm_model = st.selectbox(
                "LLM Model",
                options=MODEL_OPTIONS,
                index=MODEL_OPTIONS.index(settings["model_name"]) if settings["model_name"] in MODEL_OPTIONS else 0,
            )
            embedding_model = st.selectbox(
                "Embedding Model",
                options=EMBEDDING_OPTIONS,
                index=EMBEDDING_OPTIONS.index(settings.get("embedding_model", DEFAULT_EMBEDDING_MODEL))
                if settings.get("embedding_model", DEFAULT_EMBEDDING_MODEL) in EMBEDDING_OPTIONS
                else 0,
            )
            top_k = st.slider("Top K Chunks", 1, 10, settings["top_k"], key="settings_topk")
            chunk_size = st.select_slider(
                "Chunk Size (tokens)", [256, 512, 768, 1024], settings["chunk_size"], key="settings_chunk_size"
            )
            temperature = st.slider(
                "Temperature", 0.0, 1.0, settings["temperature"], step=0.05, key="settings_temp"
            )
            if st.button("Save Custom", use_container_width=False):
                save_system_settings(
                    llm_model,
                    embedding_model,
                    top_k,
                    temperature,
                    chunk_size,
                    chunk_overlap,
                )
                st.success("Custom settings saved")
                st.rerun()
    else:
        settings = get_system_settings()
        st.caption(
            "Read-only. Admin only can edit performance/embedding settings."
        )
        st.write(
            f"- LLM: `{settings['model_name']}`\n"
            f"- Embedding: `{settings.get('embedding_model', DEFAULT_EMBEDDING_MODEL)}`\n"
            f"- Top K: `{settings['top_k']}`\n"
            f"- Temperature: `{settings['temperature']}`\n"
            f"- Chunk Size: `{settings['chunk_size']}`"
        )

    st.markdown("#### Help")
    st.write("- Use **New Chat** in sidebar to start a fresh room with new files.")
    st.write("- Use **Chats** in sidebar to continue previous room history.")
    st.write("- Performance mode and embedding model are configured in Setting & Help.")

    if st.button("Back to Chat", use_container_width=False):
        st.session_state.active_page = "chat"
        st.rerun()


# ---------------------------------------------------------------------------
# Document processing
# ---------------------------------------------------------------------------

def process_documents(
    uploaded_files,
    chunk_size: int,
    top_k: int,
    model_name: str,
    embedding_model: str,
    temperature: float,
) -> None:
    settings = get_system_settings()
    chunk_overlap = settings["chunk_overlap"]
    user_id = st.session_state.user["id"]
    user_chroma_root = os.path.join(CHROMA_DIR, f"user_{user_id}")
    user_chroma_dir = os.path.join(user_chroma_root, f"idx_{uuid4().hex[:12]}")

    _reset_runtime_state(keep_room=True)

    print("[DeepDoc - AI-Powered Document Intelligence] Process Documents requested")
    print(
        "[DeepDoc - AI-Powered Document Intelligence] System settings:",
        {
            "model_name": settings.get("model_name"),
            "embedding_model": settings.get("embedding_model"),
            "top_k": settings.get("top_k"),
            "temperature": settings.get("temperature"),
            "chunk_size": settings.get("chunk_size"),
            "chunk_overlap": settings.get("chunk_overlap"),
        },
    )
    print(
        "[DeepDoc - AI-Powered Document Intelligence] Effective process params:",
        {
            "uploaded_files": [uf.name for uf in uploaded_files],
            "embedding_model": embedding_model,
            "top_k": top_k,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "chroma_dir": user_chroma_dir,
        },
    )

    os.makedirs(user_chroma_root, exist_ok=True)
    os.makedirs(user_chroma_dir, exist_ok=True)

    all_chunks = []
    per_file_stats: list[dict] = []
    file_records: list[dict] = []
    
    progress_bar = st.progress(0, text="Processing files...")
    for idx, uf in enumerate(uploaded_files):
        progress_bar.progress((idx + 1) / len(uploaded_files), text=f"Processing {uf.name}...")
        file_path = save_uploaded_file(uf, UPLOAD_DIR)
        chunks = process_pdf(file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks.extend(chunks)
        per_file_stats.append(
            {
                "file_name": uf.name,
                "saved_path": file_path,
                "chunk_count": len(chunks),
            }
        )
        file_records.append({"file_name": uf.name, "file_path": file_path})
    progress_bar.empty()

    room_title = (
        uploaded_files[0].name if len(uploaded_files) == 1
        else f"{uploaded_files[0].name} +{len(uploaded_files) - 1} more"
    )
    room_id = create_study_room(
        user_id=user_id,
        title=room_title,
        chroma_dir=user_chroma_dir,
        model_name=model_name,
        embedding_model=embedding_model,
        top_k=top_k,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        temperature=temperature,
    )
    for f in file_records:
        add_room_file(room_id, f["file_name"], f["file_path"])

    st.session_state.retriever = build_hybrid_retriever(
        all_chunks,
        persist_dir=user_chroma_dir,
        top_k=top_k,
        embedding_model=embedding_model,
    )
    st.session_state.current_room_id = room_id
    st.session_state.processed = True
    st.session_state.chat_history = []
    st.session_state.quiz_questions = []
    st.session_state.quiz_active = False
    st.session_state.quiz_idx = 0
    st.session_state.process_trace = {
        "uploaded_files": [uf.name for uf in uploaded_files],
        "per_file_stats": per_file_stats,
        "total_files": len(uploaded_files),
        "total_chunks": len(all_chunks),
        "model_name": model_name,
        "embedding_model": embedding_model,
        "top_k": top_k,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "bm25_weight": 0.4,
        "vector_weight": 0.6,
        "chroma_dir": user_chroma_dir,
        "room_id": room_id,
        "room_title": room_title,
    }

    for entry in os.listdir(user_chroma_root):
        old_path = os.path.join(user_chroma_root, entry)
        if old_path == user_chroma_dir:
            continue
        if os.path.isdir(old_path):
            try:
                shutil.rmtree(old_path)
            except PermissionError:
                pass

    print(
        "[DeepDoc - AI-Powered Document Intelligence] Processing complete:",
        {
            "total_chunks": len(all_chunks),
            "processed": st.session_state.processed,
            "room_id": room_id,
        },
    )


# ---------------------------------------------------------------------------
# Tab 1: Chat
# ---------------------------------------------------------------------------

def chat_tab(model_name: str, temperature: float) -> None:
    user_id = st.session_state.user["id"]
    username = st.session_state.user.get("username", "unknown")
    room_id = st.session_state.get("current_room_id")
    ready = bool(st.session_state.processed)

    if not ready:
        st.info("Upload and process a PDF to start chatting.")

    bubble_group = st.container(border=True)
    with bubble_group:
        for msg in st.session_state.chat_history:
            display_chat_message(msg["role"], msg["content"])

        if st.session_state.chat_history:
            if st.button("Clear Chat History", key="clear_chat_history"):
                clear_user_history(user_id, mode="chat", room_id=room_id)
                st.session_state.chat_history = []
                st.rerun()
        else:
            st.caption("No messages yet.")
            if ready:
                st.markdown("** Try asking:**")
                suggestions = [
                    "What is the main topic of this document?",
                    "Summarize the key points",
                    "What are the important concepts?"
                ]
                cols = st.columns(3)
                for idx, suggestion in enumerate(suggestions):
                    with cols[idx]:
                        if st.button(suggestion, key=f"suggest_{idx}", use_container_width=True):
                            st.session_state["suggested_query"] = suggestion
                            st.rerun()

    question = st.chat_input("Ask anything about the document… (Ctrl+Enter to send)", disabled=not ready)
    
    # Handle suggested query
    if "suggested_query" in st.session_state:
        question = st.session_state.pop("suggested_query")
    
    if question and ready:
        print(
            "[DeepDoc - AI-Powered Document Intelligence][Chat] User question:",
            {
                "user_id": user_id,
                "username": username,
                "model": model_name,
                "temperature": temperature,
                "question_len": len(question),
                "question": question,
            },
        )

        with bubble_group:
            display_chat_message("user", question)
        st.session_state.chat_history.append({"role": "user", "content": question})
        save_message(user_id, "chat", "user", question, room_id=room_id)

        with st.spinner("Thinking…"):
            docs = st.session_state.retriever.invoke(question)
            context = "\n\n---\n\n".join(d.page_content for d in docs)

            debug_info = getattr(st.session_state.retriever, "last_debug_info", None)
            if isinstance(debug_info, dict):
                print(
                    "[DeepDoc - AI-Powered Document Intelligence][Retrieval] Trace:",
                    {
                        "user_id": user_id,
                        "username": username,
                        "original_query": debug_info.get("original_query"),
                        "normalized_query": debug_info.get("normalized_query"),
                        "bm25_count": debug_info.get("bm25_count"),
                        "vector_count": debug_info.get("vector_count"),
                        "fused_count": debug_info.get("fused_count"),
                        "bm25_weight": debug_info.get("bm25_weight"),
                        "vector_weight": debug_info.get("vector_weight"),
                    },
                )

            chain = build_chat_chain_with_context(
                model=model_name,
                temperature=temperature,
                base_url=OLLAMA_BASE_URL,
            )
            answer = chain.invoke({"context": context, "question": question})

        debug_info = getattr(st.session_state.retriever, "last_debug_info", None)
        with st.expander("BM25 + Vector Trace", expanded=False):
            if isinstance(debug_info, dict):
                st.markdown("**Step 1 · Query**")
                st.write(f"- Original: `{debug_info.get('original_query', '')}`")
                st.write(f"- Normalized: `{debug_info.get('normalized_query', '')}`")
                bm25_tokens = debug_info.get("bm25_query_tokens", [])
                if isinstance(bm25_tokens, list):
                    st.write(f"- BM25 n-gram tokens: `{len(bm25_tokens)}`")
                    st.caption(
                        "Sample: " + ", ".join(str(tok) for tok in bm25_tokens[:20])
                    )

                st.markdown("**Step 2 · Retrieve**")
                st.write(
                    f"- BM25 docs: `{debug_info.get('bm25_count', 0)}` · "
                    f"Vector docs: `{debug_info.get('vector_count', 0)}`"
                )
                st.write(
                    f"- Weights => BM25: `{debug_info.get('bm25_weight', 0)}` · "
                    f"Vector: `{debug_info.get('vector_weight', 0)}`"
                )

                st.markdown("**Step 3 · Fuse (RRF)**")
                st.write(f"- Final chunks for LLM: `{debug_info.get('fused_count', 0)}`")

                st.markdown("**Top Fused Chunk Previews**")
                for i, preview in enumerate(debug_info.get("fused_top_previews", []), start=1):
                    if isinstance(preview, dict):
                        src = preview.get("source", "unknown")
                        page = preview.get("page")
                        text = preview.get("preview", "")
                        page_str = f" · p.{page}" if page else ""
                        st.write(f"{i}. [{src}{page_str}] {text}")
                    else:
                        st.write(f"{i}. {preview}")
            else:
                st.caption("No retrieval trace available for this question.")

            st.markdown("**Step 4 · Context Stats**")
            st.write(f"- Context chars sent to LLM: `{len(context)}`")
            st.write(f"- Model: `{model_name}` · Temperature: `{temperature}`")

        print(
            "[DeepDoc - AI-Powered Document Intelligence][Chat] Assistant response:",
            {
                "user_id": user_id,
                "username": username,
                "answer_len": len(answer),
                "answer_preview": answer[:300],
            },
        )

        with bubble_group:
            display_chat_message("assistant", answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        save_message(user_id, "chat", "assistant", answer, room_id=room_id)
        if room_id is not None:
            touch_study_room(room_id)
        st.rerun()


# ---------------------------------------------------------------------------
# Tab 2: Quiz
# ---------------------------------------------------------------------------

def quiz_tab(model_name: str, temperature: float) -> None:
    user_id = st.session_state.user["id"]
    room_id = st.session_state.get("current_room_id")

    if not st.session_state.processed:
        st.info("Upload and process a PDF to start a quiz.")
        return

    if not st.session_state.quiz_active:
        st.markdown("Generate study questions from the document, then answer them one by one.")
        col1, col2 = st.columns([3, 1])
        with col1:
            num_q = st.number_input("Number of questions", min_value=1, max_value=10, value=3)
        with col2:
            st.write("")
            st.write("")
            gen = st.button("Generate", use_container_width=True, type="primary")

        if gen:
            with st.spinner("Generating questions from document…"):
                docs = st.session_state.retriever.invoke(
                    "key concepts main ideas important facts"
                )
                context = "\n\n---\n\n".join(d.page_content for d in docs)
                st.session_state.quiz_context = context

                chain = build_quiz_chain(model_name, temperature, OLLAMA_BASE_URL)
                raw = chain.invoke({"context": context, "num_questions": num_q})
                questions = parse_quiz_questions(raw)

            if not questions:
                st.warning("Could not parse questions. Try a different model or chunk size.")
                return

            st.session_state.quiz_questions = questions
            st.session_state.quiz_idx = 0
            st.session_state.quiz_active = True
            st.rerun()
    else:
        questions = st.session_state.quiz_questions
        idx   = st.session_state.quiz_idx
        total = len(questions)

        st.progress(idx / total, text=f"Question {idx + 1} of {total}")
        st.markdown(f"### {questions[idx]}")

        student_answer = st.text_area(
            "Your Answer",
            key=f"answer_{idx}",
            placeholder="Type your answer here…",
            height=120,
        )

        col_submit, col_skip, col_quit = st.columns([2, 1, 1])
        with col_submit:
            submit = st.button("Submit Answer", type="primary", use_container_width=True)
        with col_skip:
            skip = st.button("Skip", use_container_width=True)
        with col_quit:
            quit_quiz = st.button("End Quiz", use_container_width=True)

        if quit_quiz:
            st.session_state.quiz_active = False
            st.rerun()

        if skip:
            if idx + 1 < total:
                st.session_state.quiz_idx += 1
                st.rerun()
            else:
                st.success("You have reached the end of the quiz!")
                if st.button("Start New Quiz"):
                    st.session_state.quiz_active = False
                    st.rerun()

        if submit:
            if not student_answer.strip():
                st.warning("Please write an answer before submitting.")
            else:
                with st.spinner("Evaluating your answer…"):
                    eval_chain = build_eval_chain(model_name, OLLAMA_BASE_URL)
                    feedback = eval_chain.invoke({
                        "question": questions[idx],
                        "student_answer": student_answer,
                        "context": st.session_state.quiz_context,
                    })

                save_message(user_id, "quiz", "user", questions[idx], room_id=room_id)
                save_message(user_id, "quiz", "assistant", feedback, room_id=room_id)
                if room_id is not None:
                    touch_study_room(room_id)

                st.markdown("---")
                st.markdown("#### Feedback")
                st.markdown(feedback)
                st.markdown("---")

                if idx + 1 < total:
                    if st.button("Next Question →", type="primary"):
                        st.session_state.quiz_idx += 1
                        st.rerun()
                else:
                    st.success("You have completed all questions!")
                    if st.button("Start New Quiz"):
                        st.session_state.quiz_active = False
                        st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="DeepDoc - AI-Powered Document Intelligence",
        page_icon="D",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_styles()
    initialize_state()

    # ── Auth gate ─────────────────────────────────────────────────────────
    if not is_logged_in():
        if st.session_state.auth_page == "register":
            render_register_page()
        else:
            render_login_page()
        return

    # ── Authenticated ─────────────────────────────────────────────────────
    st.markdown(
        (
            '<div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">'
            f"{FILE_AI_ICON_SVG}"
            '<h1 style="margin:0;font-size:2rem;">DeepDoc - AI-Powered Document Intelligence</h1>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )
    st.caption(
        "Chat with your PDFs · Study with AI-generated quizzes · "
        "Powered by Ollama + BM25 + Vector Search"
    )

    render_sidebar()

    if st.session_state.get("active_page") == "settings":
        render_settings_page()
        return

    settings = get_system_settings()
    model_name = settings["model_name"]
    embedding_model = settings.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
    top_k = settings["top_k"]
    temperature = settings["temperature"]
    chunk_size = settings["chunk_size"]

    with st.expander(
        "Upload & Process PDF",
        expanded=True,
    ):
        col_upload, col_settings = st.columns([2, 1])

        with col_settings:
            model_name, embedding_model, top_k, temperature, chunk_size = render_upload_settings_panel()

        with col_upload:
            uploaded_files = st.file_uploader(
                "Upload one or more PDF files",
                type=["pdf"],
                accept_multiple_files=True,
                key=f"upload_files_{st.session_state.uploader_nonce}",
            )

            if uploaded_files:
                st.markdown(f"**{len(uploaded_files)} file(s) selected**")
                for uf in uploaded_files:
                    with st.expander(f"📄 {uf.name}", expanded=False):
                        temp_path = save_uploaded_file(uf, UPLOAD_DIR)
                        preview = get_pdf_preview(temp_path)
                        if preview["success"]:
                            st.caption(f"Pages: {preview['pages']}")
                            st.text_area("Preview", preview["preview"], height=150, disabled=True)
                        else:
                            st.warning(f"Could not preview: {preview.get('error', 'Unknown error')}")

                if st.button("Process Documents", type="primary", use_container_width=True):
                    with st.spinner("Loading PDFs → Chunking → Building BM25 + Vector index…"):
                        process_documents(
                            uploaded_files,
                            chunk_size,
                            top_k,
                            model_name,
                            embedding_model,
                            temperature,
                        )
                    st.success(f"{len(uploaded_files)} file(s) indexed and ready!")
                    st.rerun()

        process_trace = st.session_state.get("process_trace")
        if isinstance(process_trace, dict):
            with st.expander("Processing Trace", expanded=False):
                st.markdown("**Step 1 · Input Files**")
                st.write(f"- Total files: `{process_trace.get('total_files', 0)}`")
                for i, file_name in enumerate(process_trace.get("uploaded_files", []), start=1):
                    st.write(f"{i}. {file_name}")

                st.markdown("**Step 2 · Chunking**")
                total_chunks = process_trace.get('total_chunks', 0)
                st.write(f"- Total chunks: `{total_chunks}`")
                for item in process_trace.get("per_file_stats", []):
                    chunk_count = item.get('chunk_count')
                    percentage = (chunk_count / total_chunks * 100) if total_chunks > 0 else 0
                    st.write(f"- {item.get('file_name')}: `{chunk_count}` chunks ({percentage:.1f}%)")

                st.markdown("**Step 3 · Index Build (Hybrid Retrieval)**")
                st.write(f"- Top K: `{process_trace.get('top_k', 0)}`")
                st.write(f"- Chunk Size: `{process_trace.get('chunk_size', 0)}`")
                st.write(f"- Chunk Overlap: `{process_trace.get('chunk_overlap', 0)}`")
                st.write(
                    f"- BM25 Weight: `{process_trace.get('bm25_weight', 0)}` · "
                    f"Vector Weight: `{process_trace.get('vector_weight', 0)}`"
                )
                st.write(f"- Chroma DB: `{process_trace.get('chroma_dir', '')}`")
                st.write(f"- Active Model: `{process_trace.get('model_name', '')}`")
                st.write(f"- Embedding Model: `{process_trace.get('embedding_model', '')}`")

    tab_chat, tab_quiz = st.tabs(["Chat", "Quiz"])

    with tab_chat:
        chat_tab(model_name, temperature)

    with tab_quiz:
        quiz_tab(model_name, temperature)


if __name__ == "__main__":
    main()

