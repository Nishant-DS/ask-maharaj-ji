"""Generate and validate chunk metadata in English."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from app.config import Settings
from app.services.ai.gemini_client import GeminiClient


class ChunkMetadata(BaseModel):
    """Metadata persisted as JSONB for one semantic chunk."""

    summary: str = Field(min_length=1)
    keywords: list[str]
    answerable_questions: list[str]


class MetadataGenerator:
    """Load the external prompt template and request structured English metadata."""

    def __init__(self, client: GeminiClient, settings: Settings) -> None:
        self._client = client
        self._model = settings.llm_model
        self._prompt_template = (Path(__file__).parents[2] / "prompts" / "metadata_prompt.txt").read_text(
            encoding="utf-8"
        )

    def generate(self, chunk_text: str) -> dict[str, object]:
        response = self._client.generate_json(self._model, self._prompt_template.format(chunk_text=chunk_text))
        try:
            return ChunkMetadata.model_validate(json.loads(response)).model_dump()
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"Gemini returned invalid metadata JSON: {error}") from error
