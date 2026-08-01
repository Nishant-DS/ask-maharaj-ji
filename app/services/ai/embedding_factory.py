"""Factory for selecting an embedding implementation from configuration."""

from app.config import Settings
from app.services.ai.base_embedding_provider import BaseEmbeddingProvider
from app.services.ai.gemini_client import GeminiClient
from app.services.ai.gemini_embedding_provider import GeminiEmbeddingProvider
from app.services.ai.jina_embedding_provider import JinaEmbeddingProvider


class EmbeddingFactory:
    """Construct the configured embedding adapter."""

    @staticmethod
    def create(settings: Settings) -> BaseEmbeddingProvider:
        if settings.embedding_provider == "jina":
            return JinaEmbeddingProvider(settings)
        if settings.embedding_provider == "gemini":
            return GeminiEmbeddingProvider(GeminiClient(settings), settings.embedding_model)
        raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")
