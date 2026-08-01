import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from app.services.youtube_playlist_service import (
    PlaylistIngestionService,
    PlaylistVideo,
    YouTubeTranscriptDownloader,
)


class FakePlaylistSource:
    def list_videos(self, playlist_url: str) -> tuple[str, list[PlaylistVideo]]:
        assert playlist_url == "https://www.youtube.com/playlist?list=PL123"
        return "Teachings", [
            PlaylistVideo("video-one", "https://www.youtube.com/watch?v=video-one", "First"),
            PlaylistVideo("video-two", "https://www.youtube.com/watch?v=video-two", "Second"),
        ]


class FakeTranscriptDownloader:
    def download(self, video_id: str, destination: Path, languages: list[str]) -> None:
        if video_id == "video-two":
            raise ValueError("Transcript unavailable")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=["start", "end", "text"])
            writer.writeheader()
            writer.writerow({"start": 0, "end": 1, "text": "राम"})


class ThreeVideoPlaylistSource:
    def list_videos(self, playlist_url: str) -> tuple[str, list[PlaylistVideo]]:
        return "Teachings", [
            PlaylistVideo("one", "https://www.youtube.com/watch?v=one", "One"),
            PlaylistVideo("two", "https://www.youtube.com/watch?v=two", "Two"),
            PlaylistVideo("three", "https://www.youtube.com/watch?v=three", "Three"),
        ]


class SuccessfulTranscriptDownloader:
    def download(self, video_id: str, destination: Path, languages: list[str]) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("start,end,text\n0,1,text\n", encoding="utf-8")


@dataclass
class RecordedCall:
    transcript: Path
    video_id: str
    title: str
    speaker: str
    language: str
    replace: bool
    dry_run: bool


class FakeIngestionService:
    def __init__(self) -> None:
        self.calls: list[RecordedCall] = []

    def ingest(self, transcript: Path, video: object, *, replace: bool, dry_run: bool) -> object:
        self.calls.append(RecordedCall(
            transcript, video.youtube_video_id, video.title, video.speaker, video.language, replace, dry_run
        ))
        return object()


def test_playlist_ingestion_downloads_then_ingests_each_video_and_continues_after_failure(tmp_path: Path) -> None:
    ingestion = FakeIngestionService()
    service = PlaylistIngestionService(FakePlaylistSource(), FakeTranscriptDownloader(), ingestion)  # type: ignore[arg-type]

    result = service.ingest_playlist(
        "https://www.youtube.com/playlist?list=PL123", tmp_path / "transcripts", speaker="Maharaj Ji",
        languages=["hi", "en"], replace=True, dry_run=False,
    )

    assert result.playlist_title == "Teachings"
    assert result.videos_discovered == 2
    assert len(result.ingested) == 1
    assert result.failures[0].video_id == "video-two"
    assert ingestion.calls == [RecordedCall(
        tmp_path / "transcripts" / "video-one.csv", "video-one", "First", "Maharaj Ji", "hi", True, False
    )]
    assert (tmp_path / "transcripts" / "video-one.csv").is_file()


def test_youtube_transcript_downloader_writes_parser_compatible_csv(tmp_path: Path, monkeypatch: object) -> None:
    class FetchedTranscript:
        def to_raw_data(self) -> list[dict[str, object]]:
            return [{"start": 1.5, "duration": 2.25, "text": "महाराज जी"}]

    class FakeApi:
        def fetch(self, video_id: str, languages: list[str]) -> FetchedTranscript:
            assert (video_id, languages) == ("abc", ["hi", "en"])
            return FetchedTranscript()

    monkeypatch.setitem(sys.modules, "youtube_transcript_api", SimpleNamespace(YouTubeTranscriptApi=FakeApi))
    destination = tmp_path / "abc.csv"
    YouTubeTranscriptDownloader().download("abc", destination, ["hi", "en"])

    with destination.open(encoding="utf-8-sig", newline="") as source:
        assert list(csv.DictReader(source)) == [{"start": "1.5", "end": "3.75", "text": "महाराज जी"}]


def test_playlist_dry_run_processes_only_the_first_two_and_reports_their_csvs(tmp_path: Path) -> None:
    ingestion = FakeIngestionService()
    service = PlaylistIngestionService(  # type: ignore[arg-type]
        ThreeVideoPlaylistSource(), SuccessfulTranscriptDownloader(), ingestion
    )

    result = service.ingest_playlist(
        "https://www.youtube.com/playlist?list=PL123", tmp_path, speaker="Maharaj Ji", languages=["hi"],
        dry_run=True,
    )

    assert result.videos_discovered == 3
    assert result.videos_processed == 2
    assert [call.video_id for call in ingestion.calls] == ["one", "two"]
    assert result.transcript_paths == (tmp_path / "one.csv", tmp_path / "two.csv")


def test_playlist_max_videos_limits_a_full_run(tmp_path: Path) -> None:
    ingestion = FakeIngestionService()
    service = PlaylistIngestionService(  # type: ignore[arg-type]
        ThreeVideoPlaylistSource(), SuccessfulTranscriptDownloader(), ingestion
    )

    result = service.ingest_playlist(
        "https://www.youtube.com/playlist?list=PL123", tmp_path, speaker="Maharaj Ji", languages=["hi"],
        max_videos=2,
    )

    assert result.videos_discovered == 3
    assert result.videos_processed == 2
    assert [call.video_id for call in ingestion.calls] == ["one", "two"]
