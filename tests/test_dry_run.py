from pathlib import Path

from app.config import Settings
from app.services.ingestion_service import IngestionService


class FakeEmbeddings:
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 1024 for _ in texts]


def test_dry_run_needs_no_database(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("start,end,text\n0,2,Hello.\n", encoding="utf-8")
    settings = Settings(_env_file=None, jina_api_key="key", google_api_key="key", metadata_generation_enabled=False)
    service = IngestionService(None, None, FakeEmbeddings(), None, settings)
    summary = service.ingest(source, dry_run=True)
    assert summary.dry_run and summary.chunks_created == 1 and summary.database_seconds is None
    assert summary.preview_chunks[0].chunk_text == "Hello."
