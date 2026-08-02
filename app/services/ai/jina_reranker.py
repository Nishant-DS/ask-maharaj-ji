"""Jina reranker API adapter with the same resilient request behavior as embeddings."""

from __future__ import annotations

import logging
from typing import Any

import httpx
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from app.config import Settings
from app.services.ai.base_reranker import BaseReranker, RerankedDocument, RerankerError
from app.services.ai.jina_embedding_provider import _is_transient

logger = logging.getLogger(__name__)


class JinaReranker(BaseReranker):
    """Call Jina's reranking endpoint and retain the original candidate indexes."""

    endpoint = "https://api.jina.ai/v1/rerank"

    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        settings.validate_ai_requirements()
        assert settings.jina_api_key is not None
        self._api_key = settings.jina_api_key.get_secret_value()
        self._model = settings.reranker_model
        self._timeout = settings.request_timeout_seconds
        self._max_retries = settings.max_retries
        self._client = client or httpx.Client(timeout=self._timeout)

    def _request(self, query: str, documents: list[str], top_n: int) -> list[RerankedDocument]:
        response = self._client.post(
            self.endpoint,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self._model, "query": query, "documents": documents, "top_n": top_n,
                "return_documents": False,
            },
        )
        response.raise_for_status()
        data: Any = response.json().get("results")
        if not isinstance(data, list):
            raise RerankerError("Jina response did not contain a 'results' reranking list.")
        try:
            return [RerankedDocument(index=int(item["index"]), relevance_score=float(item["relevance_score"]))
                    for item in data]
        except (KeyError, TypeError, ValueError) as error:
            raise RerankerError("Jina response has an invalid reranking format.") from error

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[RerankedDocument]:
        if not documents:
            return []
        if not 1 <= top_n <= len(documents):
            raise ValueError("top_n must be between 1 and the number of reranking documents")
        logger.info("Requesting Jina reranking", extra={"model": self._model, "candidate_count": len(documents)})
        try:
            retrying = Retrying(
                stop=stop_after_attempt(self._max_retries), wait=wait_exponential(min=1, max=8),
                retry=retry_if_exception(_is_transient), reraise=True,
            )
            results = retrying(self._request, query, documents, top_n)
        except httpx.HTTPStatusError as error:
            status = error.response.status_code
            if status in {401, 403}:
                raise RerankerError("Jina authentication failed. Check JINA_API_KEY.") from error
            raise RerankerError(f"Jina reranking request failed with HTTP {status}.") from error
        except (httpx.TimeoutException, httpx.NetworkError) as error:
            raise RerankerError("Jina reranking API is unreachable after retries. Check network connectivity.") from error
        if len(results) > top_n or any(item.index < 0 or item.index >= len(documents) for item in results):
            raise RerankerError("Jina response contains invalid reranking indexes.")
        return results
