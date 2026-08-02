"""Semantic retrieval orchestration for the Phase 2 read path."""

from __future__ import annotations

from app.database.repository import TranscriptChunkRepository
from app.models.retrieval import RetrievedChunk, RetrievalQuery
from app.services.ai.base_embedding_provider import BaseEmbeddingProvider
from app.services.ai.base_reranker import BaseReranker


class RetrievalService:
    """Retrieve vector candidates, then use a cross-encoder to select final chunks."""

    candidate_limit = 20

    def __init__(self, repository: TranscriptChunkRepository, embedding_provider: BaseEmbeddingProvider,
                 reranker: BaseReranker) -> None:
        self._repository = repository
        self._embedding_provider = embedding_provider
        self._reranker = reranker

    def retrieve(self, request: RetrievalQuery) -> list[RetrievedChunk]:
        """Return the top 3–5 reranked chunks, optionally scoped to a single video."""
        embedding = self._embedding_provider.embed_batch([request.query])[0]
        candidates = self._repository.find_nearest_chunks(
            embedding=embedding,
            limit=self.candidate_limit,
            youtube_video_id=request.youtube_video_id,
        )
        if not candidates:
            return []
        reranked = self._reranker.rerank(
            request.query, [chunk.chunk_text for chunk in candidates], min(request.limit, len(candidates))
        )
        return [candidates[item.index].model_copy(update={"reranker_score": item.relevance_score}) for item in reranked]
