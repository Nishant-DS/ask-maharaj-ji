"""Embedding provider port."""

from abc import ABC, abstractmethod


class EmbeddingProviderError(RuntimeError):
    """Raised when an embedding provider cannot produce valid vectors."""


class BaseEmbeddingProvider(ABC):
    """Provider-neutral batch embedding interface."""

    @abstractmethod
    def embed_batch(self, texts: list[str], task: str = "retrieval.passage") -> list[list[float]]:
        """Embed a batch of original-language texts."""
