"""Download YouTube playlist captions and pass them to the ingestion pipeline."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

from app.services.ingestion_service import IngestionService, IngestionSummary, VideoMetadata

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlaylistVideo:
    """The metadata needed to fetch and ingest one item from a playlist."""

    video_id: str
    url: str
    title: str


@dataclass(frozen=True)
class PlaylistIngestionFailure:
    """A recoverable failure for one playlist item."""

    video_id: str
    title: str
    error: str


@dataclass(frozen=True)
class PlaylistIngestionSummary:
    """Result of a playlist run, including failures that did not stop later videos."""

    playlist_title: str
    videos_discovered: int
    videos_processed: int
    ingested: tuple[IngestionSummary, ...]
    failures: tuple[PlaylistIngestionFailure, ...]
    transcript_paths: tuple[Path, ...]


class PlaylistSource(Protocol):
    def list_videos(self, playlist_url: str) -> tuple[str, list[PlaylistVideo]]: ...


class TranscriptDownloader(Protocol):
    def download(self, video_id: str, destination: Path, languages: list[str]) -> None: ...


class YtDlpPlaylistSource:
    """Resolve a playlist without downloading its audio or video."""

    def list_videos(self, playlist_url: str) -> tuple[str, list[PlaylistVideo]]:
        parsed = urlparse(playlist_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("playlist_url must be an absolute HTTP(S) URL")
        try:
            from yt_dlp import YoutubeDL
        except ImportError as error:
            raise RuntimeError("yt-dlp is required for playlist ingestion. Install project dependencies.") from error

        options: dict[str, Any] = {
            "extract_flat": "in_playlist", "skip_download": True, "quiet": True, "no_warnings": True,
        }
        with YoutubeDL(options) as downloader:
            info = downloader.extract_info(playlist_url, download=False)
        if not info or not info.get("entries"):
            raise ValueError("No videos found in the supplied playlist")
        videos = [
            PlaylistVideo(
                video_id=str(entry["id"]),
                url=str(entry.get("webpage_url") or f"https://www.youtube.com/watch?v={entry['id']}"),
                title=str(entry.get("title") or entry["id"]),
            )
            for entry in info["entries"]
            if entry and entry.get("id")
        ]
        if not videos:
            raise ValueError("No available videos found in the supplied playlist")
        return str(info.get("title") or "Untitled playlist"), videos


class YouTubeTranscriptDownloader:
    """Fetch official or auto-generated YouTube captions and write ingestion CSVs."""

    def download(self, video_id: str, destination: Path, languages: list[str]) -> None:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError as error:
            raise RuntimeError(
                "youtube-transcript-api is required for playlist ingestion. Install project dependencies."
            ) from error
        transcript = YouTubeTranscriptApi().fetch(video_id, languages=languages)
        rows = transcript.to_raw_data()
        if not rows:
            raise ValueError("YouTube returned an empty transcript")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8-sig", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=["start", "end", "text"])
            writer.writeheader()
            for row in rows:
                start, duration = float(row["start"]), float(row["duration"])
                writer.writerow({"start": start, "end": start + duration, "text": row["text"]})


class PlaylistIngestionService:
    """Download every available playlist transcript, then ingest each CSV sequentially."""

    def __init__(self, playlist_source: PlaylistSource, transcript_downloader: TranscriptDownloader,
                 ingestion_service: IngestionService) -> None:
        self._playlist_source = playlist_source
        self._transcript_downloader = transcript_downloader
        self._ingestion_service = ingestion_service

    def ingest_playlist(self, playlist_url: str, transcript_directory: Path, *, speaker: str,
                        languages: list[str], replace: bool = False, dry_run: bool = False,
                        max_videos: int | None = None) -> PlaylistIngestionSummary:
        if not speaker.strip():
            raise ValueError("speaker must not be empty")
        if not languages:
            raise ValueError("languages must include at least one language code")
        if max_videos is not None and max_videos < 1:
            raise ValueError("max_videos must be at least 1")
        playlist_title, videos = self._playlist_source.list_videos(playlist_url)
        selected_videos = videos[:max_videos] if max_videos is not None else videos
        if dry_run:
            selected_videos = selected_videos[:2]
        ingested: list[IngestionSummary] = []
        failures: list[PlaylistIngestionFailure] = []
        transcript_paths: list[Path] = []
        for position, video in enumerate(selected_videos, start=1):
            transcript_path = transcript_directory / f"{video.video_id}.csv"
            try:
                logger.info("Processing playlist video %d/%d id=%s", position, len(selected_videos), video.video_id)
                self._transcript_downloader.download(video.video_id, transcript_path, languages)
                transcript_paths.append(transcript_path)
                summary = self._ingestion_service.ingest(
                    transcript_path,
                    VideoMetadata(youtube_video_id=video.video_id, youtube_url=video.url, title=video.title,
                                  speaker=speaker, language=languages[0]),
                    replace=replace,
                    dry_run=dry_run,
                )
                ingested.append(summary)
            except Exception as error:
                logger.exception("Playlist video failed id=%s", video.video_id)
                failures.append(PlaylistIngestionFailure(video.video_id, video.title, str(error)))
        return PlaylistIngestionSummary(
            playlist_title, len(videos), len(selected_videos), tuple(ingested), tuple(failures),
            tuple(transcript_paths),
        )
