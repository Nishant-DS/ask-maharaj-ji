"""Persistence operations for transcript chunks."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from uuid import uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Date, Integer, MetaData, Table, Text, insert, select, delete
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Session

from app.models.chunk import PersistedChunk


metadata = MetaData()
transcript_chunks = Table(
    "transcript_chunks", metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column("youtube_video_id", Text, nullable=False), Column("youtube_url", Text, nullable=False),
    Column("title", Text, nullable=False), Column("speaker", Text, nullable=False),
    Column("language", Text), Column("discourse_date", Date), Column("chunk_index", Integer, nullable=False),
    Column("start_second", Integer, nullable=False), Column("end_second", Integer, nullable=False),
    Column("chunk_text", Text, nullable=False), Column("metadata", JSONB), Column("embedding", Vector()),
)


class TranscriptChunkRepository:
    """Repository with transactions but no ingestion policy."""

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self._session_factory = session_factory

    def video_exists(self, youtube_video_id: str) -> bool:
        with self._session_factory() as session:
            return session.execute(select(transcript_chunks.c.id).where(
                transcript_chunks.c.youtube_video_id == youtube_video_id).limit(1)).first() is not None

    def delete_video(self, youtube_video_id: str) -> None:
        with self._session_factory() as session, session.begin():
            session.execute(delete(transcript_chunks).where(
                transcript_chunks.c.youtube_video_id == youtube_video_id))

    def insert_chunks(self, video: object, chunks: Sequence[PersistedChunk]) -> None:
        """Atomically persist chunks for a video metadata object."""
        rows = [{
            "id": uuid4(), "youtube_video_id": video.youtube_video_id, "youtube_url": str(video.youtube_url),
            "title": video.title, "speaker": video.speaker, "language": video.language,
            "discourse_date": video.discourse_date, "chunk_index": chunk.chunk_index,
            "start_second": chunk.start_second, "end_second": chunk.end_second,
            "chunk_text": chunk.chunk_text, "metadata": chunk.metadata, "embedding": chunk.embedding,
        } for chunk in chunks]
        with self._session_factory() as session, session.begin():
            session.execute(insert(transcript_chunks), rows)
