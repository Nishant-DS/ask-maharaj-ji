from pathlib import Path

from app.services.chunking import SemanticChunker
from app.services.transcript_parser import TranscriptParser
from app.services.transcript_reconstructor import TranscriptReconstructor


def test_parser_and_chunker_preserve_rows(tmp_path: Path) -> None:
    source = tmp_path / "transcript.csv"
    source.write_text("start,end,text\n0,2,पहला वाक्य।\n2,4,Second sentence.\n", encoding="utf-8")
    segments = TranscriptParser().parse(source)
    chunks = SemanticChunker(chunk_size=20, overlap=5).chunk(segments)
    assert len(segments) == 2
    assert chunks[0].start_second == 0
    assert chunks[-1].end_second == 4
    assert "पहला वाक्य" in " ".join(chunk.chunk_text for chunk in chunks)


def test_reconstructor_merges_subtitles_and_removes_word_overlap(tmp_path: Path) -> None:
    source = tmp_path / "transcript.csv"
    source.write_text(
        "start,end,text\n18.4,19.2,हमारा अच्छा व्यवहार किसी\n19.2,20.4,किसी से धन्यवाद लेने के लिए नहीं होना चाहिए।\n",
        encoding="utf-8",
    )
    reconstructed = TranscriptReconstructor().reconstruct(TranscriptParser().parse(source))
    assert reconstructed[0].text == "हमारा अच्छा व्यवहार किसी से धन्यवाद लेने के लिए नहीं होना चाहिए।"
    assert (reconstructed[0].row_start, reconstructed[0].row_end) == (1, 2)
    assert (reconstructed[0].start_second, reconstructed[0].end_second) == (18.4, 20.4)
