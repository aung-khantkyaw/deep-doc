"""LLM Chains for Chat, Quiz Generation, and Answer Evaluation via Ollama."""
from __future__ import annotations

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.retrievers import BaseRetriever


def _format_docs(docs: list) -> str:
    return "\n\n---\n\n".join(d.page_content for d in docs)


def _build_ollama_llm(
    model: str,
    temperature: float,
    base_url: str,
    *,
    task: str,
) -> ChatOllama:
    """Create task-tuned Ollama chat model settings.

    These limits reduce generation latency in local CPU/GPU setups while
    keeping enough room for grounded answers.
    """
    common_kwargs = {
        "model": model,
        "temperature": temperature,
        "base_url": base_url,
        # Keep model warm to avoid repeated cold-start latency.
        "keep_alive": "30m",
    }

    if task == "chat":
        # Prevent incomplete answers for explain/how-style questions.
        return ChatOllama(**common_kwargs, num_ctx=2048, num_predict=420)
    if task == "quiz":
        return ChatOllama(**common_kwargs, num_ctx=2048, num_predict=700)
    if task == "eval":
        return ChatOllama(**common_kwargs, num_ctx=1536, num_predict=220)

    return ChatOllama(**common_kwargs)


# ---------------------------------------------------------------------------
# 1. Chat Chain — RAG Q&A over uploaded documents
# ---------------------------------------------------------------------------

_CHAT_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful study assistant.

Use the context below as your primary source and give a concise, direct answer.
If the context is partially relevant, provide the best grounded answer from it.
Only say "I cannot find the answer in this document." when the context is empty
or clearly unrelated to the question.
Prefer short bullets/paragraphs and end with a complete sentence.
Keep the answer under ~220 words unless the user explicitly asks for details.

Context:
{context}

Question: {question}

Answer:"""
)


def build_chat_chain(
    retriever: BaseRetriever,
    model: str,
    temperature: float,
    base_url: str,
):
    """RAG chain: retrieve relevant chunks → generate a grounded answer."""
    llm = _build_ollama_llm(model, temperature, base_url, task="chat")
    return (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | _CHAT_PROMPT
        | llm
        | StrOutputParser()
    )


def build_chat_chain_with_context(model: str, temperature: float, base_url: str):
    """Chat chain that accepts pre-retrieved context.

    Expected input: {"context": str, "question": str}
    """
    llm = _build_ollama_llm(model, temperature, base_url, task="chat")
    return _CHAT_PROMPT | llm | StrOutputParser()


# ---------------------------------------------------------------------------
# 2. Quiz Generation Chain — produce study questions from document context
# ---------------------------------------------------------------------------

_QUIZ_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert instructional designer creating high-value study questions.
Use only the document excerpt below and generate exactly {num_questions} questions
in {quiz_type} format.

Quality goals:
- Prioritize understanding, reasoning, and application over simple copy-from-text recall.
- Focus on central ideas, cause-effect relationships, comparisons, assumptions, and implications.
- Avoid trivial wording, vague phrasing, and duplicate questions.
- Keep each question self-contained and specific enough to answer from the excerpt.

Difficulty mix:
- About 30% foundational understanding, 40% conceptual reasoning, 30% applied/analytical.
- If the excerpt is short, still avoid surface-level repetition.

Rules:
- Number each item as Q1:, Q2:, Q3:, ...
- Do NOT include answers, explanations, or extra commentary.
- If quiz_type is "Multiple Choice":
    - Each question must have exactly 4 options in this exact format:
        A) ...
        B) ...
        C) ...
        D) ...
    - Include one clearly best answer and three plausible distractors.
    - Distractors must be realistic and same-domain (not joke/obviously wrong options).
    - Do not use "All of the above" or "None of the above".
- If quiz_type is "True/False":
    - Each Qn line must be a single precise statement judged as True or False.
    - Avoid absolute words like "always"/"never" unless supported by the excerpt.
    - Do not add A/B/C/D options.

Document excerpt:
{context}

Generate exactly {num_questions} questions in {quiz_type} format:"""
)


def build_quiz_chain(model: str, temperature: float, base_url: str):
    """Generate numbered study questions from a document context string."""
    llm = _build_ollama_llm(model, temperature, base_url, task="quiz")
    return _QUIZ_PROMPT | llm | StrOutputParser()


# ---------------------------------------------------------------------------
# 3. Evaluation Chain — assess student answer and give detailed feedback
# ---------------------------------------------------------------------------

_EVAL_PROMPT = ChatPromptTemplate.from_template(
    """You are a supportive teacher evaluating a student's answer.

Question: {question}

Student's Answer: {student_answer}

Quiz Type: {quiz_type}

Reference material from the document:
{context}

Return ONLY valid JSON with this exact schema:
{{
    "verdict": "Correct|Partially Correct|Incorrect",
    "what_was_right": "string",
    "what_missing_or_wrong": "string",
    "complete_answer": "string",
    "additional_feedback": "string"
}}

Rules:
- Always include all five keys.
- For Multiple Choice and True/False:
    - If verdict is Correct: keep "what_missing_or_wrong" and "complete_answer" as empty string.
    - If verdict is Partially Correct or Incorrect: include "what_missing_or_wrong"; "complete_answer" can be empty.
- Keep responses concise and grounded in the reference material.
"""
)


def build_eval_chain(model: str, base_url: str):
    """Evaluate a student's answer against the document context."""
    llm = _build_ollama_llm(model, 0.1, base_url, task="eval")
    return _EVAL_PROMPT | llm | StrOutputParser()
