from app.config import MODEL_DIMENSIONS, Settings


def test_embedding_dimension_is_derived_from_model() -> None:
    settings = Settings(_env_file=None, jina_api_key="key", google_api_key="key")
    assert settings.embedding_dimensions == MODEL_DIMENSIONS["jina-embeddings-v3"] == 1024


def test_database_settings_are_optional_until_required() -> None:
    settings = Settings(_env_file=None, jina_api_key="key", google_api_key="key")
    assert settings.postgres_host is None
