"""Gemini embedding provider adapter."""

from __future__ import annotations

from app.config import Settings
from app.services.ai.base_embedding_provider import BaseEmbeddingProvider, EmbeddingProviderError
from app.services.ai.gemini_client import GeminiClient


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    """Use the Google SDK only through the Gemini client gateway."""

    def __init__(self, client: GeminiClient, model: str) -> None:
        self._client = client
        self._model = model

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            return self._client.embed(texts, self._model)
        except Exception as error:
            raise EmbeddingProviderError(f"Gemini embedding request failed: {error}") from error
