"""General utility helpers for the DeepDoc Streamlit app."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import streamlit as st


def save_uploaded_file(uploaded_file, upload_dir: str = "data/uploads") -> str:
    """Save a Streamlit UploadedFile to disk and return the full path."""
    Path(upload_dir).mkdir(parents=True, exist_ok=True)
    file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def display_chat_message(role: str, content: str) -> None:
    """Render a chat bubble with the correct role avatar."""
    with st.chat_message(role):
        st.markdown(content)


def parse_quiz_questions(raw_text: str, quiz_type: str = "Multiple Choose") -> list[dict[str, object]]:
    """Extract structured quiz items from LLM output.

    Returns a list of dicts:
    {
        "question": str,
        "options": list[str],
        "type": str,
    }
    """
    if not isinstance(raw_text, str):
        raw_text = str(raw_text)

    normalized_type = "True/False" if quiz_type == "True/False" else "Multiple Choose"

    def _normalize_item(question: str, options: list[str] | None = None) -> dict[str, object]:
        cleaned_question = str(question).strip()
        if normalized_type == "True/False":
            normalized_options = ["True", "False"]
        else:
            normalized_options = [str(opt).strip() for opt in (options or []) if str(opt).strip()]
        return {
            "question": cleaned_question,
            "options": normalized_options,
            "type": normalized_type,
        }

    def _try_parse_json_questions(text: str) -> list[dict[str, object]]:
        cleaned = text.strip()
        if not cleaned:
            return []

        fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.IGNORECASE | re.DOTALL)
        if fence_match:
            cleaned = fence_match.group(1).strip()

        parsed: object
        try:
            parsed = json.loads(cleaned)
        except Exception:
            return []

        raw_items: list[object]
        if isinstance(parsed, list):
            raw_items = parsed
        elif isinstance(parsed, dict):
            if isinstance(parsed.get("questions"), list):
                raw_items = parsed["questions"]
            else:
                raw_items = [parsed]
        else:
            return []

        questions_from_json: list[dict[str, object]] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue

            question = (
                item.get("question")
                or item.get("statement")
                or item.get("prompt")
                or item.get("text")
            )
            if not isinstance(question, str) or not question.strip():
                continue

            options_raw = item.get("options")
            options: list[str] = []
            if isinstance(options_raw, list):
                options = [str(opt).strip() for opt in options_raw if str(opt).strip()]
            elif isinstance(options_raw, dict):
                for key in ["A", "B", "C", "D", "a", "b", "c", "d"]:
                    value = options_raw.get(key)
                    if value is not None and str(value).strip():
                        options.append(str(value).strip())

            questions_from_json.append(_normalize_item(question=question, options=options))

        return [q for q in questions_from_json if q["question"]]

    json_questions = _try_parse_json_questions(raw_text)
    if json_questions:
        return json_questions

    lines = [line.strip() for line in raw_text.splitlines() if line.strip() and not line.strip().startswith("```")]
    question_pattern_q = re.compile(
        r"^(?:#+\s*)?(?:\*\*)?(?:Q(?:uestion)?\s*)?(\d+)(?:\*\*)?\s*[:.)-]\s*(.*)$",
        re.IGNORECASE,
    )
    question_pattern_number = re.compile(
        r"^(?:#+\s*)?(?:\*\*)?(\d+)(?:\*\*)?\s*[:.)-]\s*(.*)$",
        re.IGNORECASE,
    )
    question_pattern_labeled = re.compile(
        r"^(?:#+\s*)?(?:\*\*)?(?:Q(?:uestion)?)(?:\s*\d+)?(?:\*\*)?\s*[:.)-]\s*(.*)$",
        re.IGNORECASE,
    )
    option_pattern_plain = re.compile(
        r"^(?:\*\*)?([A-D])(?:\*\*)?\s*[).:\-]\s*(.+)$",
        re.IGNORECASE,
    )
    option_pattern_paren = re.compile(
        r"^(?:\*\*)?\(([A-D])\)(?:\*\*)?\s*(.+)$",
        re.IGNORECASE,
    )

    questions: list[dict[str, object]] = []
    current_question: str | None = None
    current_options: list[str] = []

    def commit_current() -> None:
        nonlocal current_question, current_options
        if current_question:
            questions.append(_normalize_item(question=current_question, options=current_options[:]))
        current_question = None
        current_options = []

    for line in lines:
        normalized_line = re.sub(r"^\s*[-*]\s+", "", line)

        q_match = (
            question_pattern_q.match(normalized_line)
            or question_pattern_number.match(normalized_line)
            or question_pattern_labeled.match(normalized_line)
        )
        if q_match:
            commit_current()
            current_question = (q_match.group(2) if q_match.lastindex and q_match.lastindex >= 2 else q_match.group(1)).strip()
            if not current_question:
                current_question = None
            continue

        if current_question is not None:
            if not current_question:
                current_question = normalized_line
                continue

            opt_match = option_pattern_plain.match(normalized_line)
            if opt_match:
                current_options.append(opt_match.group(2).strip())
                continue

            opt_match = option_pattern_paren.match(normalized_line)
            if opt_match:
                current_options.append(opt_match.group(2).strip())
                continue

            if normalized_type == "True/False":
                if current_question:
                    current_question = f"{current_question} {normalized_line}".strip()
            elif not current_options:
                current_question = f"{current_question} {normalized_line}".strip()

    if normalized_type == "True/False" and not questions and lines:
        for line in lines:
            normalized_line = re.sub(r"^\s*[-*]\s+", "", line)
            q_match = (
                question_pattern_q.match(normalized_line)
                or question_pattern_number.match(normalized_line)
                or question_pattern_labeled.match(normalized_line)
            )
            if q_match:
                statement = (q_match.group(2) if q_match.lastindex and q_match.lastindex >= 2 else q_match.group(1)).strip()
                if statement:
                    questions.append(_normalize_item(question=statement, options=[]))
            elif normalized_line:
                questions.append(_normalize_item(question=normalized_line, options=[]))
        return questions

    commit_current()
    return questions
