"""Thin orchestration layer for the Phase 1 ingestion flow."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from time import perf_counter

from pydantic import BaseModel, HttpUrl

from app.config import Settings
from app.database.db import Database
from app.database.repository import TranscriptChunkRepository
from app.models.chunk import PersistedChunk, SemanticChunk
from app.services.ai.base_embedding_provider import BaseEmbeddingProvider
from app.services.ai.metadata_generator import MetadataGenerator
from app.services.chunking import SemanticChunker
from app.services.transcript_parser import TranscriptParser

logger = logging.getLogger(__name__)


class VideoMetadata(BaseModel):
    """Video-level properties duplicated on every persisted chunk."""

    youtube_video_id: str
    youtube_url: HttpUrl
    title: str
    speaker: str
    language: str | None = None
    discourse_date: date | None = None


@dataclass(frozen=True)
class IngestionSummary:
    """Observable result of a successful ingestion or dry run."""

    transcript_name: str
    transcript_rows: int
    chunks_created: int
    embedding_dimension: int
    metadata_generated: bool
    metadata_seconds: float
    embedding_seconds: float
    database_seconds: float | None
    total_seconds: float
    dry_run: bool
    preview_chunks: tuple[SemanticChunk, ...]


class IngestionService:
    """Coordinate parser, chunker, AI adapters, and persistence dependencies."""

    def __init__(self, database: Database | None, repository: TranscriptChunkRepository | None,
                 embedding_provider: BaseEmbeddingProvider, metadata_generator: MetadataGenerator | None,
                 settings: Settings) -> None:
        self._database, self._repository = database, repository
        self._embedding_provider, self._metadata_generator, self._settings = (
            embedding_provider, metadata_generator, settings
        )

    def ingest(self, transcript_path: Path, video: VideoMetadata | None = None, *, replace: bool = False,
               dry_run: bool = False) -> IngestionSummary:
        """Run the pipeline; dry runs execute AI work but never require or write to PostgreSQL."""
        started = perf_counter()
        logger.info("Reading transcript", extra={"transcript": transcript_path.name})
        segments = TranscriptParser().parse(transcript_path)
        chunks = SemanticChunker(self._settings.chunk_size, self._settings.chunk_overlap).chunk(segments)
        logger.info("Created chunks", extra={"transcript_rows": len(segments), "chunks": len(chunks)})
        metadata_started = perf_counter()
        metadata = self._generate_metadata(chunks)
        metadata_seconds = perf_counter() - metadata_started
        embedding_started = perf_counter()
        embeddings = self._generate_embeddings([chunk.chunk_text for chunk in chunks])
        embedding_seconds = perf_counter() - embedding_started
        database_seconds: float | None = None
        if not dry_run:
            if not video or not self._database or not self._repository:
                raise ValueError("Full ingestion requires video metadata and configured database dependencies.")
            self._database.verify(self._settings.embedding_dimensions)
            if self._repository.video_exists(video.youtube_video_id):
                if not replace:
                    raise ValueError("Video already exists; use --replace to replace its stored chunks.")
                self._repository.delete_video(video.youtube_video_id)
            stored = [PersistedChunk(**chunk.model_dump(), metadata=item_metadata, embedding=vector)
                      for chunk, item_metadata, vector in zip(chunks, metadata, embeddings, strict=True)]
            insertion_started = perf_counter()
            self._repository.insert_chunks(video, stored)
            database_seconds = perf_counter() - insertion_started
        total_seconds = perf_counter() - started
        summary = IngestionSummary(
            transcript_name=transcript_path.name, transcript_rows=len(segments), chunks_created=len(chunks),
            embedding_dimension=self._settings.embedding_dimensions,
            metadata_generated=self._metadata_generator is not None, metadata_seconds=metadata_seconds,
            embedding_seconds=embedding_seconds, database_seconds=database_seconds,
            total_seconds=total_seconds, dry_run=dry_run, preview_chunks=tuple(chunks[:5]),
        )
        logger.info(
            "Completed ingestion transcript=%s rows=%d chunks=%d metadata_seconds=%.3f "
            "embedding_seconds=%.3f database_seconds=%s total_seconds=%.3f dry_run=%s",
            summary.transcript_name, summary.transcript_rows, summary.chunks_created,
            summary.metadata_seconds, summary.embedding_seconds,
            "n/a" if summary.database_seconds is None else f"{summary.database_seconds:.3f}",
            summary.total_seconds, summary.dry_run,
        )
        return summary

    def _generate_metadata(self, chunks: list[SemanticChunk]) -> list[dict[str, object] | None]:
        if self._metadata_generator is None:
            return [None] * len(chunks)
        return [self._metadata_generator.generate(chunk.chunk_text) for chunk in chunks]

    def _generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for offset in range(0, len(texts), self._settings.embedding_batch_size):
            vectors.extend(self._embedding_provider.embed_batch(texts[offset:offset + self._settings.embedding_batch_size]))
        return vectors
