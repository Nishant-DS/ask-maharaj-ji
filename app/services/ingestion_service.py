"""Thin orchestration layer for the Phase 1 ingestion flow."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from pydantic import BaseModel, HttpUrl

from app.config import Settings
from app.database.db import Database
from app.database.repository import TranscriptChunkRepository
from app.models.chunk import PersistedChunk, SemanticChunk
from app.services.ai.base_embedding_provider import BaseEmbeddingProvider
from app.services.ai.semantic_section_generator import SectionProposal, SemanticSectionGenerator
from app.services.transcript_parser import TranscriptParser
from app.services.transcript_reconstructor import TranscriptReconstructor

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
                 embedding_provider: BaseEmbeddingProvider, metadata_generator: SemanticSectionGenerator | None,
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
        reconstructed = TranscriptReconstructor().reconstruct(segments)
        metadata_started = perf_counter()
        chunks = self._create_section_records(reconstructed)
        metadata_seconds = perf_counter() - metadata_started
        logger.info("Created semantic section records", extra={"transcript_rows": len(segments), "records": len(chunks)})
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
            stored = [PersistedChunk(**chunk.model_dump(), metadata=chunk.metadata, embedding=vector)
                      for chunk, vector in zip(chunks, embeddings, strict=True)]
            insertion_started = perf_counter()
            self._repository.insert_chunks(video, stored)
            database_seconds = perf_counter() - insertion_started
        total_seconds = perf_counter() - started
        summary = IngestionSummary(
            transcript_name=transcript_path.name, transcript_rows=len(segments), chunks_created=len(chunks),
            embedding_dimension=self._settings.embedding_dimensions,
            metadata_generated=self._metadata_generator is not None, metadata_seconds=metadata_seconds,
            embedding_seconds=embedding_seconds, database_seconds=database_seconds,
            total_seconds=total_seconds, dry_run=dry_run,
            preview_chunks=tuple(chunk for chunk in chunks if chunk.record_type == "transcript")[:5],
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

    def _create_section_records(self, rows: list[object]) -> list[SemanticChunk]:
        proposals = self._metadata_generator.generate(rows) if self._metadata_generator else [SectionProposal(
            start_row=rows[0].row_start, end_row=rows[-1].row_end, topic="Transcript", summary="Transcript section",
            questions=[],
        )]
        records: list[SemanticChunk] = []
        for index, proposal in enumerate(proposals):
            selected = [row for row in rows if row.row_end >= proposal.start_row and row.row_start <= proposal.end_row]
            if not selected:
                raise ValueError("Gemini section does not cover reconstructed transcript content")
            section_id = uuid4()
            metadata = {"topic": proposal.topic, "summary": proposal.summary, "section_id": str(section_id),
                        "record_type": "transcript", "row_start": proposal.start_row, "row_end": proposal.end_row,
                        "questions": proposal.questions}
            records.append(SemanticChunk(chunk_index=index, start_second=round(selected[0].start_second),
                end_second=round(selected[-1].end_second), chunk_text=" ".join(row.text for row in selected),
                section_id=section_id, record_type="transcript", row_start=proposal.start_row, row_end=proposal.end_row,
                metadata=metadata))
            for question in proposal.questions:
                records.append(SemanticChunk(chunk_index=index, start_second=round(selected[0].start_second),
                    end_second=round(selected[-1].end_second), chunk_text=question, section_id=section_id,
                    record_type="generated_question", row_start=proposal.start_row, row_end=proposal.end_row,
                    metadata={**metadata, "record_type": "generated_question"}))
        return records

    def _generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for offset in range(0, len(texts), self._settings.embedding_batch_size):
            vectors.extend(self._embedding_provider.embed_batch(
                texts[offset:offset + self._settings.embedding_batch_size], task="retrieval.passage"))
        return vectors
