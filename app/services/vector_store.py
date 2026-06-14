from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import re
from threading import RLock

from app.core.config import settings
from app.services.embedding_client import EmbeddingClient


@dataclass(frozen=True)
class Chunk:
    user_id: str
    document_id: str
    chunk_index: int
    content: str
    embedding: list[float]


class InMemoryVectorStore:
    def __init__(self, embedding_client: EmbeddingClient) -> None:
        self._chunks: list[Chunk] = []
        self._lock = RLock()
        self._embedding_client = embedding_client

    _logger = logging.getLogger(__name__)

    def upsert_document(self, user_id: str, document_id: str, chunk_texts: list[str]) -> int:
        embeddings = self._embedding_client.embed_texts(chunk_texts)

        with self._lock:
            self._chunks = [
                chunk
                for chunk in self._chunks
                if not (chunk.user_id == user_id and chunk.document_id == document_id)
            ]
            for index, text in enumerate(chunk_texts):
                embedding = embeddings[index] if index < len(embeddings) else []
                self._chunks.append(
                    Chunk(
                        user_id=user_id,
                        document_id=document_id,
                        chunk_index=index,
                        content=text,
                        embedding=embedding,
                    )
                )
        return len(chunk_texts)

    def retrieve(self, user_id: str, question: str, document_ids: list[str] | None, top_k: int) -> list[Chunk]:
        with self._lock:
            candidates = [chunk for chunk in self._chunks if chunk.user_id == user_id]

        if document_ids:
            allowed = set(document_ids)
            candidates = [chunk for chunk in candidates if chunk.document_id in allowed]

        query_embedding = self._embedding_client.embed_query(question)
        scored = []
        for chunk in candidates:
            lexical_score = _lexical_score(question, chunk.content)
            semantic_score = _cosine_similarity(query_embedding, chunk.embedding)
            combined_score = semantic_score + (settings.retrieval_lexical_weight * lexical_score)
            scored.append((chunk, combined_score, semantic_score, lexical_score))

        ranked = sorted(scored, key=lambda item: item[1], reverse=True)

        if settings.retrieval_log_scores and ranked:
            self._logger.info(
                "Retrieval scoring user_id=%s candidates=%s min_score=%.3f top_k=%s",
                user_id,
                len(ranked),
                settings.retrieval_min_score,
                top_k,
            )
            for position, (chunk, combined, semantic, lexical) in enumerate(ranked[:top_k], start=1):
                self._logger.info(
                    "rank=%s doc=%s chunk=%s combined=%.4f semantic=%.4f lexical=%.4f",
                    position,
                    chunk.document_id,
                    chunk.chunk_index,
                    combined,
                    semantic,
                    lexical,
                )

        selected = [chunk for chunk, combined, _, _ in ranked if combined >= settings.retrieval_min_score]
        if selected:
            return selected[:top_k]

        # Fallback keeps context available when no chunk reaches the minimum score.
        return [chunk for chunk, _, _, _ in ranked[:top_k]]


def _lexical_score(question: str, content: str) -> float:
    q_tokens = _tokenize(question)
    if not q_tokens:
        return 0.0
    c_tokens = _tokenize(content)
    return len(q_tokens.intersection(c_tokens)) / float(len(q_tokens))


def _tokenize(text: str) -> set[str]:
    # Unicode-aware tokens improve matching for French words and punctuation-heavy text.
    return {token for token in re.findall(r"[^\W_]+", text.lower(), flags=re.UNICODE) if token}


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) != len(right):
        # A dimension mismatch means chunks were embedded with a different model
        # (e.g. some real, some hash fallback). Surface it instead of silently
        # scoring them as completely dissimilar.
        InMemoryVectorStore._logger.warning(
            "Embedding dimension mismatch in similarity: query=%s chunk=%s. Re-ingest documents to fix.",
            len(left),
            len(right),
        )
        return 0.0

    numerator = sum(l * r for l, r in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)

