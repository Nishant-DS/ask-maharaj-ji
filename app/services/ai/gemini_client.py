"""The sole gateway to the official Google GenAI SDK."""

from __future__ import annotations

import logging
from typing import Any

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import Settings

logger = logging.getLogger(__name__)


class GeminiClient:
    """Encapsulate authentication, timeouts, retries, and error logging for Gemini."""

    def __init__(self, settings: Settings) -> None:
        if not settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY is required for Gemini")
        self._client = genai.Client(api_key=settings.google_api_key.get_secret_value())
        self._timeout_ms = int(settings.request_timeout_seconds * 1000)

    @retry(wait=wait_exponential(min=1, max=8), stop=stop_after_attempt(3), reraise=True)
    def generate_json(self, model: str, prompt: str) -> str:
        try:
            response = self._client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json", http_options=types.HttpOptions(timeout=self._timeout_ms)
                ),
            )
            if not response.text:
                raise RuntimeError("Gemini returned an empty response")
            return response.text
        except Exception:
            logger.exception("Gemini metadata request failed")
            raise

    @retry(wait=wait_exponential(min=1, max=8), stop=stop_after_attempt(3), reraise=True)
    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        try:
            response = self._client.models.embed_content(
                model=model,
                contents=texts,
                config=types.EmbedContentConfig(http_options=types.HttpOptions(timeout=self._timeout_ms)),
            )
            return [list(item.values) for item in response.embeddings]
        except Exception:
            logger.exception("Gemini embedding request failed")
            raise
