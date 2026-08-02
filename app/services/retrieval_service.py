"""Semantic retrieval orchestration for the Phase 2 read path."""

from __future__ import annotations

from app.database.repository import TranscriptChunkRepository
from app.models.retrieval import RetrievedChunk, RetrievalQuery
from app.services.ai.base_embedding_provider import BaseEmbeddingProvider


class RetrievalService:
    """Retrieve sections by matching transcript text and generated question aliases."""

    candidate_limit = 50

    def __init__(self, repository: TranscriptChunkRepository, embedding_provider: BaseEmbeddingProvider,
                 reranker: object | None = None) -> None:
        self._repository = repository
        self._embedding_provider = embedding_provider

    def retrieve(self, request: RetrievalQuery) -> list[RetrievedChunk]:
        """Return original transcript rows for the highest-scoring matching sections."""
        embedding = self._embedding_provider.embed_batch([request.query], task="retrieval.query")[0]
        return self._repository.find_nearest_chunks(
            embedding=embedding,
            limit=request.limit,
            youtube_video_id=request.youtube_video_id,
        )
