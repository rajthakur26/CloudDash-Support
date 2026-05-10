"""
RAG Retrieval Pipeline
- Ingests KB articles from JSON into ChromaDB (vector store)
- Hybrid retrieval: vector similarity + BM25 keyword search
- Query rewriting using conversation context
- Returns ranked, deduplicated KB chunks with relevance scores
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

from agents.models import KBCitation
from config.logging_config import get_logger
from config.settings import get_settings

logger = get_logger(__name__)
settings = get_settings()

KB_PATH = Path(__file__).parent.parent / "knowledge_base" / "articles.json"
COLLECTION_NAME = "clouddash_kb"


# ─────────────────────────────────────────────────────────────────────────────
# Document chunking
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_article(article: dict[str, Any], chunk_size: int = 500, overlap: int = 100) -> list[dict[str, Any]]:
    """Split article content into overlapping chunks. Short articles stay as one chunk."""
    content: str = article["content"]
    words = content.split()
    chunks = []

    if len(words) <= chunk_size // 4:   # Short article: keep whole
        chunks.append({
            "id": f"{article['id']}-chunk-0",
            "article_id": article["id"],
            "title": article["title"],
            "category": article["category"],
            "tags": ", ".join(article.get("tags", [])),
            "applies_to": ", ".join(article.get("applies_to", [])),
            "content": content,
            "chunk_index": 0,
        })
        return chunks

    # Word-level sliding window
    step = max(1, (chunk_size - overlap) // 4)
    i = 0
    chunk_idx = 0
    while i < len(words):
        window = words[i: i + chunk_size // 4]
        chunk_text = " ".join(window)
        chunks.append({
            "id": f"{article['id']}-chunk-{chunk_idx}",
            "article_id": article["id"],
            "title": article["title"],
            "category": article["category"],
            "tags": ", ".join(article.get("tags", [])),
            "applies_to": ", ".join(article.get("applies_to", [])),
            "content": chunk_text,
            "chunk_index": chunk_idx,
        })
        i += step
        chunk_idx += 1
        if i + step >= len(words):
            break

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# KB Retriever class
# ─────────────────────────────────────────────────────────────────────────────

class KBRetriever:
    def __init__(self) -> None:
        self._client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None
        self._bm25: BM25Okapi | None = None
        self._chunks: list[dict[str, Any]] = []
        self._tokenized_corpus: list[list[str]] = []

    # ── Initialisation ────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Load or build the vector index and BM25 index."""
        logger.info("Initializing KB retriever", persist_dir=settings.chroma_persist_dir)

        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        self._client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )

        articles = self._load_articles()
        self._chunks = []
        for article in articles:
            self._chunks.extend(_chunk_article(article))

        # Ingest into ChromaDB if collection is empty
        if self._collection.count() == 0:
            logger.info("Ingesting KB articles into ChromaDB", chunk_count=len(self._chunks))
            self._ingest_chunks()
        else:
            logger.info("ChromaDB collection already populated", count=self._collection.count())

        # Build BM25 index over chunks
        self._tokenized_corpus = [c["content"].lower().split() for c in self._chunks]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info("BM25 index built", corpus_size=len(self._chunks))

    def _load_articles(self) -> list[dict[str, Any]]:
        with open(KB_PATH, "r", encoding="utf-8") as f:
            articles = json.load(f)
        logger.info("Loaded KB articles", count=len(articles))
        return articles

    def _ingest_chunks(self) -> None:
        ids = [c["id"] for c in self._chunks]
        documents = [c["content"] for c in self._chunks]
        metadatas = [
            {
                "article_id": c["article_id"],
                "title": c["title"],
                "category": c["category"],
                "tags": c["tags"],
                "applies_to": c["applies_to"],
                "chunk_index": c["chunk_index"],
            }
            for c in self._chunks
        ]
        self._collection.add(ids=ids, documents=documents, metadatas=metadatas)

    # ── Retrieval ─────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        bm25_weight: float = 0.3,
        vector_weight: float = 0.7,
        min_score: float = 0.2,
    ) -> list[dict[str, Any]]:
        """
        Hybrid retrieval: combine vector similarity and BM25 scores.
        Returns list of result dicts sorted by combined score (descending).
        """
        if self._collection is None or self._bm25 is None:
            raise RuntimeError("KBRetriever not initialized. Call initialize() first.")

        # ── Vector search ─────────────────────────────────────────────
        vector_results = self._collection.query(
            query_texts=[query],
            n_results=min(top_k * 2, self._collection.count()),
        )

        vector_scores: dict[str, float] = {}
        vector_docs: dict[str, dict] = {}

        if vector_results["ids"] and vector_results["ids"][0]:
            for i, chunk_id in enumerate(vector_results["ids"][0]):
                # Chroma returns distance (lower = better for cosine), convert to similarity
                distance = vector_results["distances"][0][i]
                similarity = max(0.0, 1.0 - distance)
                vector_scores[chunk_id] = similarity
                vector_docs[chunk_id] = {
                    "id": chunk_id,
                    "content": vector_results["documents"][0][i],
                    "metadata": vector_results["metadatas"][0][i],
                }

        # ── BM25 search ───────────────────────────────────────────────
        tokenized_query = query.lower().split()
        bm25_raw = self._bm25.get_scores(tokenized_query)

        # Normalise BM25 scores to [0, 1]
        max_bm25 = max(bm25_raw) if max(bm25_raw) > 0 else 1.0
        bm25_norm = [s / max_bm25 for s in bm25_raw]

        bm25_scores: dict[str, float] = {}
        for i, chunk in enumerate(self._chunks):
            bm25_scores[chunk["id"]] = bm25_norm[i]

        # ── Merge scores ──────────────────────────────────────────────
        all_ids = set(vector_scores.keys()) | set(bm25_scores.keys())
        merged: list[dict[str, Any]] = []

        for cid in all_ids:
            vs = vector_scores.get(cid, 0.0)
            bs = bm25_scores.get(cid, 0.0)
            combined = vector_weight * vs + bm25_weight * bs

            if combined < min_score:
                continue

            # Get doc info: prefer vector_docs, fallback to chunks list
            if cid in vector_docs:
                doc = vector_docs[cid]
            else:
                chunk = next((c for c in self._chunks if c["id"] == cid), None)
                if chunk is None:
                    continue
                doc = {
                    "id": cid,
                    "content": chunk["content"],
                    "metadata": {
                        "article_id": chunk["article_id"],
                        "title": chunk["title"],
                        "category": chunk["category"],
                        "tags": chunk["tags"],
                        "applies_to": chunk["applies_to"],
                        "chunk_index": chunk["chunk_index"],
                    },
                }

            merged.append({
                "chunk_id": cid,
                "article_id": doc["metadata"]["article_id"],
                "title": doc["metadata"]["title"],
                "category": doc["metadata"]["category"],
                "content": doc["content"],
                "vector_score": vs,
                "bm25_score": bs,
                "combined_score": combined,
            })

        # Deduplicate by article_id (keep highest-scoring chunk per article)
        best_per_article: dict[str, dict] = {}
        for item in merged:
            aid = item["article_id"]
            if aid not in best_per_article or item["combined_score"] > best_per_article[aid]["combined_score"]:
                best_per_article[aid] = item

        results = sorted(best_per_article.values(), key=lambda x: x["combined_score"], reverse=True)[:top_k]

        logger.info(
            "KB retrieval complete",
            query_preview=query[:80],
            results_count=len(results),
            top_score=results[0]["combined_score"] if results else 0,
        )
        return results

    def rewrite_query(self, query: str, conversation_summary: str) -> str:
        """
        Simple context-aware query rewriting.
        Prepends key entities extracted from conversation summary to the query.
        A more sophisticated version would call the LLM for rewriting.
        """
        if not conversation_summary:
            return query
        # Prepend summary context to enrich the semantic search
        return f"{conversation_summary.strip()} {query.strip()}"

    def format_citations(self, results: list[dict[str, Any]]) -> list[KBCitation]:
        return [
            KBCitation(
                article_id=r["article_id"],
                title=r["title"],
                category=r["category"],
                relevance_score=round(r["combined_score"], 3),
            )
            for r in results
        ]

    def format_context_for_prompt(self, results: list[dict[str, Any]]) -> str:
        """Format retrieved chunks into a context block for the LLM prompt."""
        if not results:
            return "No relevant knowledge base articles found."

        parts = []
        for r in results:
            parts.append(
                f"[{r['article_id']}] {r['title']}\n"
                f"Category: {r['category']}\n"
                f"Content: {r['content']}\n"
            )
        return "\n---\n".join(parts)

    @property
    def article_count(self) -> int:
        if self._collection:
            return self._collection.count()
        return 0


# ── Singleton ─────────────────────────────────────────────────────────────────
_retriever: KBRetriever | None = None


def get_retriever() -> KBRetriever:
    global _retriever
    if _retriever is None:
        _retriever = KBRetriever()
        _retriever.initialize()
    return _retriever
