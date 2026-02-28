"""Hybrid Retriever: BM25 (n-gram term matching) + Vector (semantic).

Uses a custom implementation of Reciprocal Rank Fusion (RRF) to merge
BM25 and vector search results without depending on EnsembleRetriever.
"""
from __future__ import annotations

import re
from typing import Any

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

# Lightweight embedding model — runs fully locally, no API key needed.
EMBED_MODEL = "all-MiniLM-L6-v2"


class HybridRetriever(BaseRetriever):
    """Reciprocal Rank Fusion of BM25 + vector retrievers."""

    bm25_retriever: Any = None
    vector_retriever: Any = None
    bm25_weight: float = 0.4
    vector_weight: float = 0.6
    k: int = 4
    last_debug_info: dict[str, Any] | None = None

    def _normalize_query(self, query: str) -> str:
        """Light normalization without changing user intent."""
        cleaned = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", query.lower())).strip()
        return cleaned or query

    def _clean_preview(self, text: str) -> str:
        """Improve readability for PDFs extracted as spaced characters."""
        compact = re.sub(r"\s+", " ", text).strip()
        tokens = compact.split(" ")
        if len(tokens) >= 20:
            sample = tokens[:80]
            single_ratio = sum(1 for t in sample if len(t) == 1) / len(sample)
            if single_ratio > 0.55:
                compact = "".join(tokens)
        return compact

    def _doc_preview(self, doc: Document) -> dict[str, Any]:
        preview_text = self._clean_preview(doc.page_content)[:220]
        return {
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page"),
            "preview": preview_text,
        }

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        """Retrieve from both, fuse with weighted RRF, return top-k."""
        normalized_query = self._normalize_query(query)
        bm25_docs = self.bm25_retriever.invoke(normalized_query)
        vec_docs = self.vector_retriever.invoke(query)

        rrf_k = 60  # standard RRF constant
        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for rank, doc in enumerate(bm25_docs):
            key = doc.page_content
            doc_map[key] = doc
            scores[key] = scores.get(key, 0.0) + self.bm25_weight / (rrf_k + rank + 1)

        for rank, doc in enumerate(vec_docs):
            key = doc.page_content
            doc_map[key] = doc
            scores[key] = scores.get(key, 0.0) + self.vector_weight / (rrf_k + rank + 1)

        sorted_keys = sorted(scores, key=scores.get, reverse=True)  # type: ignore[arg-type]
        fused_docs = [doc_map[k] for k in sorted_keys[: self.k]]

        self.last_debug_info = {
            "original_query": query,
            "normalized_query": normalized_query,
            "bm25_weight": self.bm25_weight,
            "vector_weight": self.vector_weight,
            "bm25_count": len(bm25_docs),
            "vector_count": len(vec_docs),
            "fused_count": len(fused_docs),
            "bm25_top_previews": [self._doc_preview(d) for d in bm25_docs[:3]],
            "vector_top_previews": [self._doc_preview(d) for d in vec_docs[:3]],
            "fused_top_previews": [self._doc_preview(d) for d in fused_docs[:3]],
        }

        return fused_docs


def build_hybrid_retriever(
    chunks: list[Document],
    persist_dir: str = "data/chroma_db",
    top_k: int = 4,
    bm25_weight: float = 0.4,
    vector_weight: float = 0.6,
) -> HybridRetriever:
    """Build a hybrid BM25 + vector retriever.

    BM25 uses n-gram term frequency scoring (good for exact keyword matches).
    Vector search uses sentence-transformers embeddings (good for semantic matches).
    Results are fused using Reciprocal Rank Fusion (RRF).

    Args:
        chunks: Chunked Document list from processor.py.
        persist_dir: ChromaDB persistence directory.
        top_k: Number of chunks each retriever returns per query.
        bm25_weight: Weight for BM25 results in fusion (0.0 – 1.0).
        vector_weight: Weight for vector results in fusion (0.0 – 1.0).

    Returns:
        HybridRetriever ready to invoke with a string query.
    """
    # 1. Vector store (ChromaDB + sentence-transformers)
    embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=persist_dir,
    )
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    # 2. BM25 retriever (Okapi BM25 — n-gram term statistics)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = top_k

    # 3. Fuse both with custom RRF
    return HybridRetriever(
        bm25_retriever=bm25_retriever,
        vector_retriever=vector_retriever,
        bm25_weight=bm25_weight,
        vector_weight=vector_weight,
        k=top_k,
    )
