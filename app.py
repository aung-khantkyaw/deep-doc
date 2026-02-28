"""DeepDoc — NotebookLM-style PDF study assistant with user auth.

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
    init_db,
    save_message,
    get_user_history,
    clear_user_history,
)
from auth.manager import (
    register,
    login,
    create_default_admin,
    get_system_settings,
    save_system_settings,
)
from engine.processor import process_pdf
from engine.retriever import build_hybrid_retriever
from engine.llm_chain import (
    build_chat_chain,
    build_chat_chain_with_context,
    build_quiz_chain,
    build_eval_chain,
)
from utils.helpers import save_uploaded_file, display_chat_message, parse_quiz_questions

load_dotenv()

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
UPLOAD_DIR: str      = os.getenv("UPLOAD_DIR", "data/uploads")
CHROMA_DIR: str      = os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")

MODEL_OPTIONS = [
    "llama3.1:8b",
    "llama3.2:3b",
    "mistral:7b",
    "phi3:mini",
    "phi4:14b",
    "gemma2:9b",
    "qwen2.5:7b",
    "deepseek-r1:7b",
]

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
        # RAG
        "chat_history": [],
        "retriever": None,
        "processed": False,
        "quiz_questions": [],
        "quiz_idx": 0,
        "quiz_active": False,
        "quiz_context": "",
        "process_trace": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def is_logged_in() -> bool:
    return st.session_state.user is not None


def is_admin() -> bool:
    return is_logged_in() and st.session_state.user["role"] == "admin"


# ---------------------------------------------------------------------------
# Auth pages
# ---------------------------------------------------------------------------

def render_login_page() -> None:
    col = st.columns([1, 2, 1])[1]
    with col:
        st.markdown("## 📄 DeepDoc")
        st.caption("Sign in to continue")
        st.markdown("---")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login", type="primary", use_container_width=True):
            ok, msg, user = login(username, password)
            if ok:
                st.session_state.user = user
                st.session_state.chat_history = [
                    {"role": r["role"], "content": r["content"]}
                    for r in get_user_history(user["id"], mode="chat")
                ]
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
        st.markdown("## 📄 DeepDoc — Register")
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
        if st.button("← Back to Login", use_container_width=True):
            st.session_state.auth_page = "login"
            st.rerun()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> tuple[str, int, float, int]:
    user = st.session_state.user
    settings = get_system_settings()

    fast_model = "llama3.2:3b"
    quality_model = "llama3.1:8b"

    st.sidebar.title("📄 DeepDoc")
    st.sidebar.markdown(f"👤 **{user['username']}** `{user['role']}`")

    if st.sidebar.button("Logout", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.sidebar.markdown("---")

    # ── Admin: editable settings ─────────────────────────────────────────
    if is_admin():
        st.sidebar.subheader("⚙️ System Settings")

        fast_preset = {
            "model_name": fast_model,
            "top_k": 3,
            "temperature": 0.2,
            "chunk_size": 512,
        }
        quality_preset = {
            "model_name": quality_model,
            "top_k": 5,
            "temperature": 0.3,
            "chunk_size": 768,
        }

        if settings["model_name"] == fast_model:
            default_mode = "Fast"
        elif settings["model_name"] == quality_model:
            default_mode = "Quality"
        else:
            default_mode = "Custom"

        performance_mode = st.sidebar.radio(
            "Performance Mode",
            options=["Fast", "Quality", "Custom"],
            index=["Fast", "Quality", "Custom"].index(default_mode),
            help="Fast: quicker responses on CPU · Quality: better answer quality · Custom: choose any model",
        )

        if performance_mode == "Fast":
            model_name = fast_preset["model_name"]
            top_k = fast_preset["top_k"]
            temperature = fast_preset["temperature"]
            chunk_size = fast_preset["chunk_size"]
            chunk_overlap = settings["chunk_overlap"]

            needs_apply = (
                settings["model_name"] != model_name
                or settings["top_k"] != top_k
                or settings["temperature"] != temperature
                or settings["chunk_size"] != chunk_size
            )
            if needs_apply:
                save_system_settings(model_name, top_k, temperature, chunk_size, chunk_overlap)
                settings = get_system_settings()
                st.sidebar.success("Fast preset auto-applied.")

            st.sidebar.write(f"- Model: `{model_name}`")
            st.sidebar.write(f"- Top K: `{top_k}`")
            st.sidebar.write(f"- Temperature: `{temperature}`")
            st.sidebar.write(f"- Chunk Size: `{chunk_size}`")
            st.sidebar.caption("Preset is auto-applied in this mode.")
        elif performance_mode == "Quality":
            model_name = quality_preset["model_name"]
            top_k = quality_preset["top_k"]
            temperature = quality_preset["temperature"]
            chunk_size = quality_preset["chunk_size"]
            chunk_overlap = settings["chunk_overlap"]

            needs_apply = (
                settings["model_name"] != model_name
                or settings["top_k"] != top_k
                or settings["temperature"] != temperature
                or settings["chunk_size"] != chunk_size
            )
            if needs_apply:
                save_system_settings(model_name, top_k, temperature, chunk_size, chunk_overlap)
                settings = get_system_settings()
                st.sidebar.success("Quality preset auto-applied.")

            st.sidebar.write(f"- Model: `{model_name}`")
            st.sidebar.write(f"- Top K: `{top_k}`")
            st.sidebar.write(f"- Temperature: `{temperature}`")
            st.sidebar.write(f"- Chunk Size: `{chunk_size}`")
            st.sidebar.caption("Preset is auto-applied in this mode.")
        else:
            model_name = st.sidebar.selectbox(
                "LLM Model",
                options=MODEL_OPTIONS,
                index=MODEL_OPTIONS.index(settings["model_name"])
                if settings["model_name"] in MODEL_OPTIONS
                else 0,
            )
            top_k = st.sidebar.slider("Top K Chunks", 1, 10, settings["top_k"])
            chunk_size = st.sidebar.select_slider(
                "Chunk Size (tokens)", [256, 512, 768, 1024], settings["chunk_size"]
            )
            temperature = st.sidebar.slider(
                "Temperature", 0.0, 1.0, settings["temperature"], step=0.05
            )
            chunk_overlap = settings["chunk_overlap"]

            if st.sidebar.button("💾 Save Settings", use_container_width=True):
                save_system_settings(model_name, top_k, temperature, chunk_size, chunk_overlap)
                st.sidebar.success("Settings saved.")

    # ── User: read-only display ──────────────────────────────────────────
    else:
        model_name  = settings["model_name"]
        top_k       = settings["top_k"]
        temperature = settings["temperature"]
        chunk_size  = settings["chunk_size"]

        st.sidebar.subheader("⚙️ System Settings")
        if model_name == fast_model:
            mode_label = "Fast"
        elif model_name == quality_model:
            mode_label = "Quality"
        else:
            mode_label = "Custom"
        st.sidebar.write(f"- Mode: `{mode_label}`")
        st.sidebar.write(f"- Model: `{model_name}`")
        st.sidebar.write(f"- Top K: `{top_k}`")
        st.sidebar.write(f"- Temperature: `{temperature}`")
        st.sidebar.write(f"- Chunk Size: `{chunk_size}`")
        st.sidebar.caption("Settings are managed by admin.")

    st.sidebar.markdown("---")
    if st.session_state.processed:
        st.sidebar.success("✅ Document loaded")
    else:
        st.sidebar.warning("⚠️ No document loaded")

    return model_name, top_k, temperature, chunk_size


# ---------------------------------------------------------------------------
# Document processing
# ---------------------------------------------------------------------------

def process_documents(uploaded_files, chunk_size: int, top_k: int) -> None:
    settings = get_system_settings()
    chunk_overlap = settings["chunk_overlap"]
    user_id = st.session_state.user["id"]
    user_chroma_root = os.path.join(CHROMA_DIR, f"user_{user_id}")
    user_chroma_dir = os.path.join(user_chroma_root, f"idx_{uuid4().hex[:12]}")

    old_retriever = st.session_state.get("retriever")
    st.session_state.retriever = None
    if old_retriever is not None:
        del old_retriever
    gc.collect()

    print("[DeepDoc] Process Documents requested")
    print(
        "[DeepDoc] System settings:",
        {
            "model_name": settings.get("model_name"),
            "top_k": settings.get("top_k"),
            "temperature": settings.get("temperature"),
            "chunk_size": settings.get("chunk_size"),
            "chunk_overlap": settings.get("chunk_overlap"),
        },
    )
    print(
        "[DeepDoc] Effective process params:",
        {
            "uploaded_files": [uf.name for uf in uploaded_files],
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
    for uf in uploaded_files:
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

    st.session_state.retriever = build_hybrid_retriever(
        all_chunks, persist_dir=user_chroma_dir, top_k=top_k
    )
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
        "model_name": settings.get("model_name"),
        "top_k": top_k,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "bm25_weight": 0.4,
        "vector_weight": 0.6,
        "chroma_dir": user_chroma_dir,
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
        "[DeepDoc] Processing complete:",
        {"total_chunks": len(all_chunks), "processed": st.session_state.processed},
    )


# ---------------------------------------------------------------------------
# Tab 1: Chat
# ---------------------------------------------------------------------------

def chat_tab(model_name: str, temperature: float) -> None:
    user_id = st.session_state.user["id"]
    username = st.session_state.user.get("username", "unknown")

    if not st.session_state.processed:
        st.info("Upload and process a PDF to start chatting.")
        return

    for msg in st.session_state.chat_history:
        display_chat_message(msg["role"], msg["content"])

    question = st.chat_input("Ask anything about the document…")
    if question:
        print(
            "[DeepDoc][Chat] User question:",
            {
                "user_id": user_id,
                "username": username,
                "model": model_name,
                "temperature": temperature,
                "question_len": len(question),
                "question": question,
            },
        )

        display_chat_message("user", question)
        st.session_state.chat_history.append({"role": "user", "content": question})
        save_message(user_id, "chat", "user", question)

        with st.spinner("Thinking…"):
            docs = st.session_state.retriever.invoke(question)
            context = "\n\n---\n\n".join(d.page_content for d in docs)

            debug_info = getattr(st.session_state.retriever, "last_debug_info", None)
            if isinstance(debug_info, dict):
                print(
                    "[DeepDoc][Retrieval] Trace:",
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
        with st.expander("🔍 BM25 + Vector Trace", expanded=False):
            if isinstance(debug_info, dict):
                st.markdown("**Step 1 · Query**")
                st.write(f"- Original: `{debug_info.get('original_query', '')}`")
                st.write(f"- Normalized: `{debug_info.get('normalized_query', '')}`")

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
            "[DeepDoc][Chat] Assistant response:",
            {
                "user_id": user_id,
                "username": username,
                "answer_len": len(answer),
                "answer_preview": answer[:300],
            },
        )

        display_chat_message("assistant", answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        save_message(user_id, "chat", "assistant", answer)

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History"):
            clear_user_history(user_id, mode="chat")
            st.session_state.chat_history = []
            st.rerun()


# ---------------------------------------------------------------------------
# Tab 2: Quiz
# ---------------------------------------------------------------------------

def quiz_tab(model_name: str, temperature: float) -> None:
    user_id = st.session_state.user["id"]

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
            gen = st.button("Generate ✨", use_container_width=True, type="primary")

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
            submit = st.button("✅ Submit Answer", type="primary", use_container_width=True)
        with col_skip:
            skip = st.button("⏭️ Skip", use_container_width=True)
        with col_quit:
            quit_quiz = st.button("🚪 End Quiz", use_container_width=True)

        if quit_quiz:
            st.session_state.quiz_active = False
            st.rerun()

        if skip:
            if idx + 1 < total:
                st.session_state.quiz_idx += 1
                st.rerun()
            else:
                st.success("🎉 You have reached the end of the quiz!")
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

                save_message(user_id, "quiz", "user",      questions[idx])
                save_message(user_id, "quiz", "assistant", feedback)

                st.markdown("---")
                st.markdown("#### 📝 Feedback")
                st.markdown(feedback)
                st.markdown("---")

                if idx + 1 < total:
                    if st.button("Next Question →", type="primary"):
                        st.session_state.quiz_idx += 1
                        st.rerun()
                else:
                    st.success("🎉 You have completed all questions!")
                    if st.button("Start New Quiz"):
                        st.session_state.quiz_active = False
                        st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="DeepDoc",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    initialize_state()

    # ── Auth gate ─────────────────────────────────────────────────────────
    if not is_logged_in():
        if st.session_state.auth_page == "register":
            render_register_page()
        else:
            render_login_page()
        return

    # ── Authenticated ─────────────────────────────────────────────────────
    st.title("📄 DeepDoc")
    st.caption(
        "Chat with your PDFs · Study with AI-generated quizzes · "
        "Powered by Ollama + BM25 + Vector Search"
    )

    model_name, top_k, temperature, chunk_size = render_sidebar()

    with st.expander(
        "📂 Upload & Process PDF",
        expanded=not st.session_state.processed,
    ):
        uploaded_files = st.file_uploader(
            "Upload one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            for uf in uploaded_files:
                st.write(f"• {uf.name}")

            if st.button("⚙️ Process Documents", type="primary", use_container_width=True):
                with st.spinner("Loading PDFs → Chunking → Building BM25 + Vector index…"):
                    process_documents(uploaded_files, chunk_size, top_k)
                st.success(f"✅ {len(uploaded_files)} file(s) indexed and ready!")
                st.rerun()

        process_trace = st.session_state.get("process_trace")
        if isinstance(process_trace, dict):
            with st.expander("🔎 Processing Trace", expanded=False):
                st.markdown("**Step 1 · Input Files**")
                st.write(f"- Total files: `{process_trace.get('total_files', 0)}`")
                for i, file_name in enumerate(process_trace.get("uploaded_files", []), start=1):
                    st.write(f"{i}. {file_name}")

                st.markdown("**Step 2 · Chunking**")
                for item in process_trace.get("per_file_stats", []):
                    st.write(f"- {item.get('file_name')}: `{item.get('chunk_count')}` chunks")
                st.write(f"- Total chunks: `{process_trace.get('total_chunks', 0)}`")

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

    tab_chat, tab_quiz = st.tabs(["💬 Chat", "🎓 Quiz"])

    with tab_chat:
        chat_tab(model_name, temperature)

    with tab_quiz:
        quiz_tab(model_name, temperature)


if __name__ == "__main__":
    main()

