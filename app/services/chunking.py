"""Semantic, row-preserving transcript chunking."""

from __future__ import annotations

import re

from app.models.chunk import SemanticChunk
from app.models.transcript import TranscriptSegment

_SENTENCE_END = re.compile(r"[.!?।！？]+(?:[\"'”’)]*)\s*$")
_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def token_count(text: str) -> int:
    """A deterministic multilingual approximation suitable for size control."""
    return len(_TOKEN.findall(text))


class SemanticChunker:
    """Create approximately token-sized chunks without ever cutting a source row."""

    def __init__(self, chunk_size: int, overlap: int) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, segments: list[TranscriptSegment]) -> list[SemanticChunk]:
        chunks: list[SemanticChunk] = []
        start = 0
        index = 0
        while start < len(segments):
            end = self._choose_end(segments, start)
            selected = segments[start:end]
            chunks.append(SemanticChunk(
                chunk_index=index,
                start_second=round(selected[0].start_second),
                end_second=round(selected[-1].end_second),
                chunk_text=" ".join(item.text for item in selected),
            ))
            index += 1
            if end == len(segments):
                break
            start = self._overlap_start(segments, start, end)
        return chunks

    def _choose_end(self, segments: list[TranscriptSegment], start: int) -> int:
        """Prefer the latest sentence boundary near the target, never excluding a long row."""
        total = 0
        last_boundary: int | None = None
        cursor = start
        while cursor < len(segments):
            row_tokens = token_count(segments[cursor].text)
            if cursor > start and total + row_tokens > self.chunk_size:
                break
            total += row_tokens
            cursor += 1
            if _SENTENCE_END.search(segments[cursor - 1].text):
                last_boundary = cursor
        return last_boundary or cursor

    def _overlap_start(self, segments: list[TranscriptSegment], start: int, end: int) -> int:
        total = 0
        cursor = end
        while cursor > start and total < self.overlap:
            cursor -= 1
            total += token_count(segments[cursor].text)
        # A one-row chunk cannot overlap itself; advancing is essential to avoid a loop.
        return cursor if start < cursor < end else end
