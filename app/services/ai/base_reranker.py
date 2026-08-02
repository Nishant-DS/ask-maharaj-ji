"""Provider-neutral reranking port."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


class RerankerError(RuntimeError):
    """Raised when a reranking provider cannot score candidate documents."""


@dataclass(frozen=True)
class RerankedDocument:
    """A candidate's original position and its provider relevance score."""

    index: int
    relevance_score: float


class BaseReranker(ABC):
    """Rank text candidates against one query."""

    @abstractmethod
    def rerank(self, query: str, documents: list[str], top_n: int) -> list[RerankedDocument]:
        """Return at most ``top_n`` candidates, ordered by descending relevance."""
