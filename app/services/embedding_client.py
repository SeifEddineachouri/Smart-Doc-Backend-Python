from __future__ import annotations

import hashlib
import logging
import math
from typing import Any

from google import genai
from google.genai import types

from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Generates embeddings via Gemini, with a deterministic local fallback.

    Falls back to a hashed local embedding only when the provider is
    unavailable. Every fallback is logged at WARNING so a misconfigured model
    name can never silently degrade retrieval to keyword matching again.
    """

    def __init__(self) -> None:
        self._client = genai.Client(api_key=settings.gemini_api_key) if settings.gemini_api_key else None
        if self._client is None:
            logger.warning(
                "No Gemini API key configured; retrieval will use the local hash embedding fallback only."
            )

    def embed_text(self, text: str) -> list[float]:
        vectors = self._embed([text], task_type=settings.embedding_query_task_type)
        return vectors[0] if vectors else self._local_embedding(text)

    def embed_query(self, text: str) -> list[float]:
        vectors = self._embed([text], task_type=settings.embedding_query_task_type)
        return vectors[0] if vectors else self._local_embedding(text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, task_type=settings.embedding_document_task_type)

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        if not texts:
            return []

        if self._client is None:
            return [self._local_embedding(text) for text in texts]

        try:
            response = self._client.models.embed_content(
                model=settings.embedding_model_name,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=settings.embedding_dimensions,
                ),
            )
            vectors = self._extract_vectors(response)
            if len(vectors) == len(texts):
                # Reduced-dimension Gemini embeddings are not pre-normalized.
                return [_normalize(vector) for vector in vectors]
            logger.warning(
                "Gemini embedding returned %s vectors for %s inputs (model=%s); using local fallback.",
                len(vectors),
                len(texts),
                settings.embedding_model_name,
            )
        except Exception as exc:  # noqa: BLE001 - keep retrieval available, but never silently
            logger.warning(
                "Gemini embedding call failed (model=%s task=%s): %s. Falling back to local hash embedding.",
                settings.embedding_model_name,
                task_type,
                exc,
            )

        return [self._local_embedding(text) for text in texts]

    def _extract_vectors(self, response: Any) -> list[list[float]]:
        embeddings = getattr(response, "embeddings", None)
        if embeddings is None and isinstance(response, dict):
            embeddings = response.get("embeddings")
        if not embeddings:
            return []

        vectors: list[list[float]] = []
        for embedding in embeddings:
            values = getattr(embedding, "values", None)
            if values is None and isinstance(embedding, dict):
                values = embedding.get("values")
            if not values:
                continue
            vectors.append([float(value) for value in values])
        return vectors

    def _local_embedding(self, text: str) -> list[float]:
        # Hashed bag-of-words fallback. Uses the SAME dimensionality as the real
        # embeddings so that cosine similarity never zeroes out on a dim mismatch.
        dimensions = max(settings.embedding_dimensions, 32)
        vector = [0.0] * dimensions
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % dimensions
            vector[index] += 1.0
        return _normalize(vector)


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]
