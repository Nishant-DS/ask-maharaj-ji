"""Transcript domain types."""

from pydantic import BaseModel, Field


class TranscriptSegment(BaseModel):
    """One immutable timed source row from a transcript."""

    start_second: float = Field(ge=0)
    end_second: float = Field(ge=0)
    text: str = Field(min_length=1)
    row_index: int = Field(ge=1)


class ReconstructedSegment(BaseModel):
    """Consecutive subtitle rows reconstructed into one complete sentence or thought."""

    start_second: float = Field(ge=0)
    end_second: float = Field(ge=0)
    row_start: int = Field(ge=1)
    row_end: int = Field(ge=1)
    text: str = Field(min_length=1)
