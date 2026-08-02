"""Reconstruct coherent text from overlapping subtitle rows without fabricating timing."""

from __future__ import annotations

import re

from app.models.transcript import ReconstructedSegment, TranscriptSegment

_SENTENCE_END = re.compile(r"[.!?।！？]+(?:[\"'”’)]*)\s*$")


def _append_without_overlap(existing: str, addition: str) -> str:
    """Remove exact word overlap commonly repeated by rolling subtitle captions."""
    existing_words, added_words = existing.split(), addition.split()
    maximum = min(len(existing_words), len(added_words))
    for size in range(maximum, 0, -1):
        if [word.casefold() for word in existing_words[-size:]] == [word.casefold() for word in added_words[:size]]:
            return " ".join(existing_words + added_words[size:])
    return " ".join(existing_words + added_words)


class TranscriptReconstructor:
    """Merge subtitle rows until a sentence boundary while retaining source-row provenance."""

    def reconstruct(self, rows: list[TranscriptSegment]) -> list[ReconstructedSegment]:
        result: list[ReconstructedSegment] = []
        current: ReconstructedSegment | None = None
        for row in rows:
            if current is None:
                current = ReconstructedSegment(start_second=row.start_second, end_second=row.end_second,
                                               row_start=row.row_index, row_end=row.row_index, text=row.text)
            else:
                current = current.model_copy(update={
                    "end_second": row.end_second, "row_end": row.row_index,
                    "text": _append_without_overlap(current.text, row.text),
                })
            if _SENTENCE_END.search(current.text):
                result.append(current)
                current = None
        if current is not None:
            result.append(current)
        return result
