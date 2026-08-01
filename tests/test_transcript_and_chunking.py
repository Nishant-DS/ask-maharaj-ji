from pathlib import Path

from app.services.chunking import SemanticChunker
from app.services.transcript_parser import TranscriptParser


def test_parser_and_chunker_preserve_rows(tmp_path: Path) -> None:
    source = tmp_path / "transcript.csv"
    source.write_text("start,end,text\n0,2,पहला वाक्य।\n2,4,Second sentence.\n", encoding="utf-8")
    segments = TranscriptParser().parse(source)
    chunks = SemanticChunker(chunk_size=20, overlap=5).chunk(segments)
    assert len(segments) == 2
    assert chunks[0].start_second == 0
    assert chunks[-1].end_second == 4
    assert "पहला वाक्य" in " ".join(chunk.chunk_text for chunk in chunks)
