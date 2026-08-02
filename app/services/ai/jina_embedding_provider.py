"""Jina embeddings API adapter with transient-error retry semantics."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import httpx
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from app.config import Settings
from app.services.ai.base_embedding_provider import BaseEmbeddingProvider, EmbeddingProviderError

logger = logging.getLogger(__name__)


def _is_transient(error: BaseException) -> bool:
    """Retry network errors and service throttling/outages, never credentials or bad input."""
    if isinstance(error, (httpx.TimeoutException, httpx.NetworkError)):
        return True
    return isinstance(error, httpx.HTTPStatusError) and error.response.status_code in {408, 429, 500, 502, 503, 504}


class JinaEmbeddingProvider(BaseEmbeddingProvider):
    """Call Jina's OpenAI-compatible embeddings endpoint in ordered batches."""

    endpoint = "https://api.jina.ai/v1/embeddings"

    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        settings.validate_ai_requirements()
        assert settings.jina_api_key is not None
        self._api_key = settings.jina_api_key.get_secret_value()
        self._model = settings.embedding_model
        self._expected_dimensions = settings.embedding_dimensions
        self._timeout = settings.request_timeout_seconds
        self._max_retries = settings.max_retries
        self._client = client or httpx.Client(timeout=self._timeout)

    def _request(self, texts: list[str], task: str) -> list[list[float]]:
        response = self._client.post(
            self.endpoint,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"model": self._model, "input": texts, "task": task},
        )
        response.raise_for_status()
        data: Any = response.json().get("data")
        if not isinstance(data, list):
            raise EmbeddingProviderError("Jina response did not contain a 'data' embedding list.")
        try:
            vectors = [item["embedding"] for item in sorted(data, key=lambda item: item["index"])]
        except (KeyError, TypeError) as error:
            raise EmbeddingProviderError("Jina response has an invalid embedding format.") from error
        return vectors

    def embed_batch(self, texts: list[str], task: str = "retrieval.passage") -> list[list[float]]:
        """Embed one batch and validate result count and fixed model dimensionality."""
        if not texts:
            return []
        logger.info("Requesting Jina embeddings", extra={"model": self._model, "batch_size": len(texts)})
        try:
            retrying = Retrying(
                stop=stop_after_attempt(self._max_retries), wait=wait_exponential(min=1, max=8),
                retry=retry_if_exception(_is_transient), reraise=True,
                before_sleep=lambda state: logger.warning(
                    "Retrying transient Jina request", extra={"attempt": state.attempt_number}
                ),
            )
            vectors = retrying(self._request, texts, task)
        except httpx.HTTPStatusError as error:
            status = error.response.status_code
            if status in {401, 403}:
                raise EmbeddingProviderError("Jina authentication failed. Check JINA_API_KEY.") from error
            try:
                detail = error.response.json().get("detail") or error.response.json().get("message")
            except ValueError:
                detail = None
            suffix = f" {detail}" if isinstance(detail, str) else ""
            raise EmbeddingProviderError(f"Jina API request failed with HTTP {status}.{suffix}") from error
        except (httpx.TimeoutException, httpx.NetworkError) as error:
            raise EmbeddingProviderError("Jina API is unreachable after retries. Check network connectivity.") from error
        if len(vectors) != len(texts):
            raise EmbeddingProviderError(f"Jina returned {len(vectors)} embeddings for {len(texts)} inputs.")
        for index, vector in enumerate(vectors):
            if not isinstance(vector, list) or len(vector) != self._expected_dimensions:
                actual = len(vector) if isinstance(vector, list) else "invalid"
                raise EmbeddingProviderError(
                    f"Jina model '{self._model}' returned vector {index} with {actual} dimensions; "
                    f"expected {self._expected_dimensions}. Check the configured model and API response."
                )
        logger.info("Received Jina embeddings", extra={"count": len(vectors), "dimensions": self._expected_dimensions})
        return vectors
