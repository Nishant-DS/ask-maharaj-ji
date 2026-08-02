"""Chunk domain types."""

from typing import Any, Literal
from uuid import UUID
from pydantic import BaseModel, Field


class SemanticChunk(BaseModel):
    """A contiguous, timestamped semantic transcript chunk."""

    chunk_index: int = Field(ge=0)
    start_second: int = Field(ge=0)
    end_second: int = Field(ge=0)
    chunk_text: str = Field(min_length=1)
    section_id: UUID | None = None
    record_type: Literal["transcript", "generated_question"] = "transcript"
    row_start: int | None = Field(default=None, ge=1)
    row_end: int | None = Field(default=None, ge=1)
    metadata: dict[str, Any] | None = None


class PersistedChunk(SemanticChunk):
    """Chunk enriched with generated metadata and its vector."""

    embedding: list[float]
