"""Strict Gemini semantic-section generation for reconstructed transcripts."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from app.config import Settings
from app.models.transcript import ReconstructedSegment
from app.services.ai.gemini_client import GeminiClient

logger = logging.getLogger(__name__)


class SectionProposal(BaseModel):
    start_row: int = Field(ge=1)
    end_row: int = Field(ge=1)
    topic: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    questions: list[str]

    @model_validator(mode="after")
    def valid_range(self) -> "SectionProposal":
        if self.end_row < self.start_row:
            raise ValueError("end_row must not precede start_row")
        return self


class SectionResponse(BaseModel):
    sections: list[SectionProposal] = Field(min_length=1)


class SemanticSectionGenerator:
    """Turn row-indexed reconstructed text into complete semantic sections."""

    def __init__(self, client: GeminiClient, settings: Settings) -> None:
        self._client, self._model = client, settings.llm_model
        self._prompt = (Path(__file__).parents[2] / "prompts" / "semantic_sections_prompt.txt").read_text(encoding="utf-8")
        self._questions_per_section = settings.questions_per_section
        self._max_tokens = settings.semantic_section_max_tokens
        self._languages = {"english": settings.generate_english, "hindi": settings.generate_hindi,
                           "roman_hindi": settings.generate_roman_hindi}

    def generate(self, rows: list[ReconstructedSegment]) -> list[SectionProposal]:
        proposals: list[SectionProposal] = []
        batch: list[ReconstructedSegment] = []
        tokens = 0
        for row in rows:
            row_tokens = len(row.text.split())
            if batch and tokens + row_tokens > self._max_tokens:
                proposals.extend(self._generate_batch(batch))
                batch, tokens = [], 0
            batch.append(row)
            tokens += row_tokens
        if batch:
            proposals.extend(self._generate_batch(batch))
        available = {source_row for row in rows for source_row in range(row.row_start, row.row_end + 1)}
        for proposal in proposals:
            if proposal.start_row not in available or proposal.end_row not in available:
                raise ValueError("Gemini section references transcript rows outside the supplied transcript")
        return proposals

    def _generate_batch(self, rows: list[ReconstructedSegment]) -> list[SectionProposal]:
        transcript = "\n".join(f"rows {row.row_start}-{row.row_end}: {row.text}" for row in rows)
        response = self._client.generate_json(self._model, self._prompt.format(
            questions_per_section=self._questions_per_section, generate_english=self._languages["english"],
            generate_hindi=self._languages["hindi"], generate_roman_hindi=self._languages["roman_hindi"], transcript=transcript,
        ))
        try:
            proposals = SectionResponse.model_validate(json.loads(response)).sections
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"Gemini returned invalid semantic-section JSON: {error}") from error
        if any(len(proposal.questions) < 2 for proposal in proposals):
            raise ValueError("Gemini must generate between 2 and 4 probable questions per semantic section")
        capped: list[SectionProposal] = []
        for proposal in proposals:
            if len(proposal.questions) > self._questions_per_section:
                logger.warning(
                    "Gemini generated too many questions; retaining the configured total",
                    extra={"returned": len(proposal.questions), "retained": self._questions_per_section},
                )
            capped.append(proposal.model_copy(update={"questions": proposal.questions[:self._questions_per_section]}))
        return capped
