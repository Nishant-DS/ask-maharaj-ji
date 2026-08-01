"""Persistence operations for transcript chunks."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any
from uuid import uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Date, Integer, MetaData, Table, Text, delete, insert, select, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Session

from app.models.chunk import PersistedChunk
from app.models.retrieval import RetrievedChunk


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

    def find_nearest_chunks(
        self, embedding: list[float], limit: int, youtube_video_id: str | None = None
    ) -> list[RetrievedChunk]:
        """Return chunks ordered by pgvector cosine distance without applying business policy."""
        query = """
            SELECT youtube_video_id, youtube_url, title, speaker, language, discourse_date,
                   chunk_index, start_second, end_second, chunk_text, metadata,
                   embedding <=> CAST(:embedding AS vector) AS cosine_distance
            FROM transcript_chunks
            WHERE embedding IS NOT NULL
        """
        params: dict[str, Any] = {"embedding": "[" + ",".join(map(str, embedding)) + "]", "limit": limit}
        if youtube_video_id:
            query += " AND youtube_video_id = :youtube_video_id"
            params["youtube_video_id"] = youtube_video_id
        query += " ORDER BY embedding <=> CAST(:embedding AS vector) LIMIT :limit"
        with self._session_factory() as session:
            rows = session.execute(text(query), params).mappings().all()
        return [RetrievedChunk.model_validate(dict(row)) for row in rows]

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
