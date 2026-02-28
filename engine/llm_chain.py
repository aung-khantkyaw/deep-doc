"""LLM Chains for Chat, Quiz Generation, and Answer Evaluation via Ollama."""
from __future__ import annotations

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.retrievers import BaseRetriever


def _format_docs(docs: list) -> str:
    return "\n\n---\n\n".join(d.page_content for d in docs)


# ---------------------------------------------------------------------------
# 1. Chat Chain — RAG Q&A over uploaded documents
# ---------------------------------------------------------------------------

_CHAT_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful study assistant.

Use the context below as your primary source and give a concise, direct answer.
If the context is partially relevant, provide the best grounded answer from it.
Only say "I cannot find the answer in this document." when the context is empty
or clearly unrelated to the question.

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
    llm = ChatOllama(model=model, temperature=temperature, base_url=base_url)
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
    llm = ChatOllama(model=model, temperature=temperature, base_url=base_url)
    return _CHAT_PROMPT | llm | StrOutputParser()


# ---------------------------------------------------------------------------
# 2. Quiz Generation Chain — produce study questions from document context
# ---------------------------------------------------------------------------

_QUIZ_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert teacher creating study questions for students.
Based on the document excerpt below, generate exactly {num_questions} \
clear, educational questions.

Rules:
- Cover the most important concepts in the text.
- Use a mix of factual, conceptual, and analytical questions.
- Number each question as Q1:, Q2:, Q3: etc.
- Do NOT include answers.

Document excerpt:
{context}

Generate {num_questions} questions:"""
)


def build_quiz_chain(model: str, temperature: float, base_url: str):
    """Generate numbered study questions from a document context string."""
    llm = ChatOllama(model=model, temperature=temperature, base_url=base_url)
    return _QUIZ_PROMPT | llm | StrOutputParser()


# ---------------------------------------------------------------------------
# 3. Evaluation Chain — assess student answer and give detailed feedback
# ---------------------------------------------------------------------------

_EVAL_PROMPT = ChatPromptTemplate.from_template(
    """You are a supportive teacher evaluating a student's answer.

Question: {question}

Student's Answer: {student_answer}

Reference material from the document:
{context}

Please provide structured feedback:
1. **Verdict**: Correct / Partially Correct / Incorrect
2. **What was right**: (what the student understood well)
3. **What was missing or wrong**: (gaps or errors)
4. **Complete answer**: (the ideal answer based on the document)

Be encouraging, precise, and educational in your response."""
)


def build_eval_chain(model: str, base_url: str):
    """Evaluate a student's answer against the document context."""
    llm = ChatOllama(model=model, temperature=0.1, base_url=base_url)
    return _EVAL_PROMPT | llm | StrOutputParser()
