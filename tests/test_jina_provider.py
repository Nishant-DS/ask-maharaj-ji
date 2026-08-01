import httpx
import pytest

from app.config import Settings
from app.services.ai.base_embedding_provider import EmbeddingProviderError
from app.services.ai.jina_embedding_provider import JinaEmbeddingProvider


def _settings() -> Settings:
    return Settings(_env_file=None, jina_api_key="test-key", google_api_key="test-key")


def test_jina_provider_validates_dimension() -> None:
    response = httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1]}]})
    client = httpx.Client(transport=httpx.MockTransport(lambda _: response))
    with pytest.raises(EmbeddingProviderError, match="1 dimensions; expected 1024"):
        JinaEmbeddingProvider(_settings(), client).embed_batch(["hello"])


def test_jina_provider_returns_ordered_vectors() -> None:
    vector = [0.0] * 1024
    response = httpx.Response(200, json={"data": [{"index": 1, "embedding": vector}, {"index": 0, "embedding": vector}]})
    client = httpx.Client(transport=httpx.MockTransport(lambda _: response))
    assert len(JinaEmbeddingProvider(_settings(), client).embed_batch(["one", "two"])) == 2
