"""General utility helpers for the DeepDoc Streamlit app."""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from pypdf import PdfReader


def save_uploaded_file(uploaded_file, upload_dir: str = "data/uploads") -> str:
    """Save a Streamlit UploadedFile to disk and return the full path."""
    Path(upload_dir).mkdir(parents=True, exist_ok=True)
    file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def get_pdf_preview(file_path: str, max_pages: int = 2) -> dict:
    """Extract preview info from PDF."""
    try:
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)
        preview_text = ""
        for i in range(min(max_pages, total_pages)):
            preview_text += reader.pages[i].extract_text() or ""
        return {
            "pages": total_pages,
            "preview": preview_text[:500],
            "success": True
        }
    except Exception as e:
        return {"pages": 0, "preview": "", "success": False, "error": str(e)}


def display_chat_message(role: str, content: str) -> None:
    """Render a chat bubble with the correct role avatar."""
    with st.chat_message(role):
        st.markdown(content)


def parse_quiz_questions(raw_text: str) -> list[str]:
    """Extract numbered questions (Q1:, Q2:, ...) from LLM output."""
    questions = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        if stripped and stripped[0] == "Q" and ":" in stripped:
            questions.append(stripped)
    return questions
