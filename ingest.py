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
from app.services.ai.semantic_section_generator import SemanticSectionGenerator
from app.services.ingestion_service import IngestionService, IngestionSummary, VideoMetadata


def configure_logging() -> None:
    """Configure consistent, structured-compatible command-line logs."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest a transcript CSV into Ask Maharaj Ji.")
    parser.add_argument("transcript", type=Path, help="CSV with start,end,text columns, or a directory of CSVs")
    parser.add_argument("--video-id", help="YouTube video ID (required for a single full-ingestion CSV)")
    parser.add_argument("--youtube-url", help="Canonical YouTube URL (required for a single full-ingestion CSV)")
    parser.add_argument("--title", help="Video title (required for a single full-ingestion CSV; optional folder prefix)")
    parser.add_argument("--speaker", help="Speaker name (required for every full ingestion)")
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


def _folder_video(path: Path, args: argparse.Namespace) -> VideoMetadata | None:
    """Derive video metadata for playlist-downloaded ``<video-id>.csv`` files."""
    if args.dry_run:
        return None
    if not args.speaker:
        raise ValueError("Directory ingestion requires --speaker. The filename is used as the YouTube video ID.")
    video_id = path.stem
    title = f"{args.title} — {video_id}" if args.title else video_id
    return VideoMetadata(
        youtube_video_id=video_id, youtube_url=f"https://www.youtube.com/watch?v={video_id}", title=title,
        speaker=args.speaker, language=args.language, discourse_date=args.discourse_date,
    )
    print(f"Transcript: {summary.transcript_name}")
    print(f"Transcript rows: {summary.transcript_rows}")
    print(f"Chunks created: {summary.chunks_created}")
    print(f"Embeddings generated: {summary.chunks_created}")
    print(f"Embedding dimension: {summary.embedding_dimension}")
    print(f"Metadata generated: {'yes' if summary.metadata_generated else 'no'}")
    if summary.preview_chunks:
        chunk = summary.preview_chunks[0]
        print("\nFirst semantic section:")
        print(f"Section ID: {chunk.section_id}")
        print(f"Rows: {chunk.row_start}–{chunk.row_end}")
        print(f"Time: {chunk.start_second}s–{chunk.end_second}s")
        if chunk.metadata:
            print(f"Topic: {chunk.metadata.get('topic', 'n/a')}")
            print(f"Summary: {chunk.metadata.get('summary', 'n/a')}")
            questions = chunk.metadata.get("questions", [])
            if questions:
                print("Generated questions:")
                for question in questions:
                    print(f"- {question}")
        print("Transcript:")
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
        metadata_generator = SemanticSectionGenerator(GeminiClient(settings), settings)
        database = repository = None
        video = None
        if not args.dry_run:
            settings.validate_database_requirements()
            database = Database(settings)
            repository = TranscriptChunkRepository(database.session_factory)
            if not args.transcript.is_dir():
                video = _video_from_args(args)
        service = IngestionService(database, repository, embedding_provider, metadata_generator, settings)
        if args.transcript.is_dir():
            csvs = sorted(path for path in args.transcript.iterdir() if path.is_file() and path.suffix.lower() == ".csv")
            if not csvs:
                raise ValueError(f"No CSV files found in directory: {args.transcript}")
            failures = 0
            for transcript_path in csvs:
                try:
                    summary = service.ingest(
                        transcript_path, _folder_video(transcript_path, args), replace=args.replace, dry_run=args.dry_run
                    )
                    if args.dry_run:
                        _print_dry_run(summary)
                except Exception as error:
                    failures += 1
                    logging.getLogger(__name__).exception("Ingestion failed for %s: %s", transcript_path, error)
            return 0 if failures == 0 else 2
        summary = service.ingest(args.transcript, video, replace=args.replace, dry_run=args.dry_run)
        if args.dry_run:
            _print_dry_run(summary)
        return 0
    except Exception as error:
        logging.getLogger(__name__).exception("Ingestion failed: %s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
