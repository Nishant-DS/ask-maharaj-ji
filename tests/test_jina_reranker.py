import json

import httpx

from app.config import Settings
from app.services.ai.jina_reranker import JinaReranker


def _settings() -> Settings:
    return Settings(_env_file=None, jina_api_key="test-key", google_api_key="test-key")


def test_jina_reranker_returns_ordered_original_indexes() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://api.jina.ai/v1/rerank"
        assert request.headers["authorization"] == "Bearer test-key"
        assert json.loads(request.content) == {
            "model": "jina-reranker-v3", "query": "question", "documents": ["first", "second"],
            "top_n": 2, "return_documents": False,
        }
        return httpx.Response(200, json={"results": [
            {"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.6},
        ]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    results = JinaReranker(_settings(), client).rerank("question", ["first", "second"], 2)
    assert [(result.index, result.relevance_score) for result in results] == [(1, 0.9), (0, 0.6)]
