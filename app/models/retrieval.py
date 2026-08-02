"""Read models returned by semantic retrieval."""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """A chunk selected by vector retrieval and optional reranking."""

    youtube_video_id: str
    youtube_url: str
    title: str
    speaker: str
    language: str | None
    discourse_date: date | None
    chunk_index: int
    start_second: int
    end_second: int
    chunk_text: str
    metadata: dict[str, Any] | None
    cosine_distance: float
    reranker_score: float | None = None


class RetrievalQuery(BaseModel):
    """Validated input for a retrieval request."""

    query: str = Field(min_length=1, max_length=2_000)
    limit: int = Field(default=5, ge=3, le=5, description="Number of reranked chunks to return")
    youtube_video_id: str | None = Field(default=None, min_length=1)
