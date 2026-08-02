"""CLI entry point for downloading and ingesting every captioned video in a YouTube playlist."""

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
from app.services.ingestion_service import IngestionService
from app.services.youtube_playlist_service import (
    PlaylistIngestionService,
    YouTubeTranscriptDownloader,
    YtDlpPlaylistSource,
)
from ingest import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and ingest every available transcript in a YouTube playlist.")
    parser.add_argument("playlist_url", help="YouTube playlist URL")
    parser.add_argument("--speaker", required=True, help="Speaker name stored with every video")
    parser.add_argument("--languages", nargs="+", default=["hi", "en"], help="Caption language preference order")
    parser.add_argument("--transcript-dir", type=Path, default=Path("transcripts/playlist"),
                        help="Directory where downloaded <video-id>.csv files are kept")
    parser.add_argument("--replace", action="store_true", help="Replace chunks already stored for each video")
    parser.add_argument("--dry-run", action="store_true", help="Download and process transcripts without PostgreSQL writes")
    parser.add_argument("--max-videos", type=int, help="Process at most this many videos in playlist order")
    return parser.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()
    try:
        settings = Settings()
        settings.validate_ai_requirements()
        database = repository = None
        if not args.dry_run:
            settings.validate_database_requirements()
            database = Database(settings)
            repository = TranscriptChunkRepository(database.session_factory)
        ingestion = IngestionService(
            database, repository, EmbeddingFactory.create(settings),
            SemanticSectionGenerator(GeminiClient(settings), settings),
            settings,
        )
        service = PlaylistIngestionService(YtDlpPlaylistSource(), YouTubeTranscriptDownloader(), ingestion)
        result = service.ingest_playlist(
            args.playlist_url, args.transcript_dir, speaker=args.speaker, languages=args.languages,
            replace=args.replace, dry_run=args.dry_run, max_videos=args.max_videos,
        )
        print(f"Playlist: {result.playlist_title}")
        print(f"Videos discovered: {result.videos_discovered}")
        print(f"Videos processed: {result.videos_processed}")
        print(f"Videos ingested: {len(result.ingested)}")
        print(f"Failures: {len(result.failures)}")
        if result.transcript_paths:
            print("Saved CSVs:")
            for transcript_path in result.transcript_paths:
                print(f"- {transcript_path.resolve()}")
        for failure in result.failures:
            print(f"- {failure.video_id} ({failure.title}): {failure.error}")
        return 0 if not result.failures else 2
    except Exception as error:
        logging.getLogger(__name__).exception("Playlist ingestion failed: %s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
