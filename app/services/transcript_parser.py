"""CSV parser isolated from downstream chunking and persistence."""

from __future__ import annotations

import csv
from pathlib import Path

from app.models.transcript import TranscriptSegment


class TranscriptParseError(ValueError):
    """Raised when a source transcript cannot be interpreted safely."""


class TranscriptParser:
    """Parse the documented start,end,text CSV format."""

    required_columns = {"start", "end", "text"}

    def parse(self, path: Path) -> list[TranscriptSegment]:
        if not path.is_file():
            raise FileNotFoundError(f"Transcript file does not exist: {path}")
        segments: list[TranscriptSegment] = []
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if not reader.fieldnames or not self.required_columns.issubset(reader.fieldnames):
                raise TranscriptParseError("CSV must include start,end,text header columns")
            for row_number, row in enumerate(reader, start=2):
                try:
                    text = (row.get("text") or "").strip()
                    segment = TranscriptSegment(
                        start_second=float(row["start"]), end_second=float(row["end"]), text=text,
                        row_index=row_number - 1,
                    )
                    if segment.end_second < segment.start_second:
                        raise TranscriptParseError("end precedes start")
                    segments.append(segment)
                except (TypeError, ValueError) as error:
                    raise TranscriptParseError(f"Invalid row {row_number}: {error}") from error
        if not segments:
            raise TranscriptParseError("Transcript CSV contains no usable rows")
        return segments
