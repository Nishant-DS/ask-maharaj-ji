"""Transcript domain types."""

from pydantic import BaseModel, Field


class TranscriptSegment(BaseModel):
    """One immutable timed source row from a transcript."""

    start_second: float = Field(ge=0)
    end_second: float = Field(ge=0)
    text: str = Field(min_length=1)
