"""PDF Loading and Text Chunking."""
from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


def load_pdf(file_path: str | Path) -> list[Document]:
    """Extract text from every page of a PDF and return as Document list."""
    reader = PdfReader(str(file_path))
    docs: list[Document] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": Path(file_path).name, "page": i + 1},
                )
            )
    return docs


def split_documents(
    docs: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 80,
) -> list[Document]:
    """Split documents into overlapping chunks for retrieval.

    Uses RecursiveCharacterTextSplitter which respects paragraph → sentence →
    word boundaries — good for preserving n-gram context used by BM25.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def process_pdf(
    file_path: str | Path,
    chunk_size: int = 512,
    chunk_overlap: int = 80,
) -> list[Document]:
    """Full pipeline: load PDF → split into chunks."""
    docs = load_pdf(file_path)
    return split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
