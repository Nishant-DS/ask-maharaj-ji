from app.models.retrieval import RetrievedChunk, RetrievalQuery
from app.services.retrieval_service import RetrievalService
from app.services.ai.base_reranker import RerankedDocument


class FakeEmbeddings:
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        assert texts == ["What is devotion?"]
        return [[0.0] * 1024]


class FakeRepository:
    def find_nearest_chunks(self, embedding: list[float], limit: int, youtube_video_id: str | None):
        assert len(embedding) == 1024
        assert limit == 20
        assert youtube_video_id == "video-1"
        return [RetrievedChunk(
            youtube_video_id="video-1", youtube_url="https://youtube.test/watch?v=video-1",
            title="Talk", speaker="Maharaj Ji", language="hi", discourse_date=None,
            chunk_index=0, start_second=0, end_second=10, chunk_text="Bhakti.",
            metadata=None, cosine_distance=0.12,
        )]


class FakeReranker:
    def rerank(self, query: str, documents: list[str], top_n: int) -> list[RerankedDocument]:
        assert query == "What is devotion?"
        assert documents == ["Bhakti."]
        assert top_n == 1
        return [RerankedDocument(index=0, relevance_score=0.98)]


def test_retrieval_embeds_query_and_returns_matches() -> None:
    service = RetrievalService(FakeRepository(), FakeEmbeddings(), FakeReranker())
    results = service.retrieve(RetrievalQuery(query="What is devotion?", youtube_video_id="video-1"))
    assert results[0].chunk_text == "Bhakti."
    assert results[0].reranker_score == 0.98
