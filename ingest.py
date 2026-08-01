"""CLI entry point for full transcript ingestion and database-free dry runs."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from app.config import Settings
from app.database.db import Database
from app.database.repository import TranscriptChunkRepository
from app.services.ai.embedding_factory import EmbeddingFactory
from app.services.ai.gemini_client import GeminiClient
from app.services.ai.metadata_generator import MetadataGenerator
from app.services.ingestion_service import IngestionService, IngestionSummary, VideoMetadata


def configure_logging() -> None:
    """Configure consistent, structured-compatible command-line logs."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest a transcript CSV into Ask Maharaj Ji.")
    parser.add_argument("transcript", type=Path, help="CSV with start,end,text columns")
    parser.add_argument("--video-id", help="YouTube video ID (required unless --dry-run)")
    parser.add_argument("--youtube-url", help="Canonical YouTube URL (required unless --dry-run)")
    parser.add_argument("--title", help="Video title (required unless --dry-run)")
    parser.add_argument("--speaker", help="Speaker name (required unless --dry-run)")
    parser.add_argument("--language", help="BCP-47 language tag, e.g. hi or en")
    parser.add_argument("--discourse-date", help="ISO date, e.g. 2025-01-31")
    parser.add_argument("--replace", action="store_true", help="Replace chunks already stored for this video")
    parser.add_argument("--dry-run", action="store_true", help="Run parsing and AI generation without PostgreSQL writes")
    return parser.parse_args()


def _video_from_args(args: argparse.Namespace) -> VideoMetadata:
    required = {"--video-id": args.video_id, "--youtube-url": args.youtube_url,
                "--title": args.title, "--speaker": args.speaker}
    missing = [flag for flag, value in required.items() if not value]
    if missing:
        raise ValueError("Full ingestion requires " + ", ".join(missing) + ". Or use --dry-run.")
    return VideoMetadata(youtube_video_id=args.video_id, youtube_url=args.youtube_url, title=args.title,
                         speaker=args.speaker, language=args.language, discourse_date=args.discourse_date)


def _print_dry_run(summary: IngestionSummary) -> None:
    print("---------------------------------")
    print(f"Transcript: {summary.transcript_name}")
    print(f"Transcript rows: {summary.transcript_rows}")
    print(f"Chunks created: {summary.chunks_created}")
    print(f"Embeddings generated: {summary.chunks_created}")
    print(f"Embedding dimension: {summary.embedding_dimension}")
    print(f"Metadata generated: {'yes' if summary.metadata_generated else 'no'}")
    if summary.preview_chunks:
        print("\nFirst 5 chunks:")
        for chunk in summary.preview_chunks:
            print(f"\n[{chunk.chunk_index}] {chunk.start_second}s–{chunk.end_second}s")
            print(chunk.chunk_text)
    print("\nDry run successful.")
    print("No database writes performed.")
    print("---------------------------------")


def main() -> int:
    configure_logging()
    args = parse_args()
    try:
        settings = Settings()
        settings.validate_ai_requirements()
        embedding_provider = EmbeddingFactory.create(settings)
        metadata_generator = MetadataGenerator(GeminiClient(settings), settings) if settings.metadata_generation_enabled else None
        database = repository = None
        video = None
        if not args.dry_run:
            settings.validate_database_requirements()
            database = Database(settings)
            repository = TranscriptChunkRepository(database.session_factory)
            video = _video_from_args(args)
        service = IngestionService(database, repository, embedding_provider, metadata_generator, settings)
        summary = service.ingest(args.transcript, video, replace=args.replace, dry_run=args.dry_run)
        if args.dry_run:
            _print_dry_run(summary)
        return 0
    except Exception as error:
        logging.getLogger(__name__).exception("Ingestion failed: %s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
