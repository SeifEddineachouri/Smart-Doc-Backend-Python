from __future__ import annotations

import hashlib
import math
from typing import Any

from google import genai

from app.core.config import settings


class EmbeddingClient:
    """Generates embeddings via Gemini, with a deterministic local fallback."""

    def __init__(self) -> None:
        self._client = genai.Client(api_key=settings.gemini_api_key) if settings.gemini_api_key else None

    def embed_text(self, text: str) -> list[float]:
        vectors = self.embed_texts([text])
        return vectors[0] if vectors else self._local_embedding(text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        if self._client is None:
            return [self._local_embedding(text) for text in texts]

        try:
            response = self._client.models.embed_content(
                model=settings.embedding_model_name,
                contents=texts,
            )
            vectors = self._extract_vectors(response)
            if len(vectors) == len(texts):
                return vectors
        except Exception:
            pass

        # Keep retrieval available even when provider embeddings fail.
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
        # Simple hashed embedding fallback for local/offline semantic ranking.
        dimensions = max(settings.local_embedding_dimensions, 32)
        vector = [0.0] * dimensions
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % dimensions
            vector[index] += 1.0

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

