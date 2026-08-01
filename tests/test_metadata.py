from unittest.mock import Mock

from app.config import Settings
from app.services.ai.metadata_generator import MetadataGenerator


def test_metadata_generator_validates_mocked_json() -> None:
    client = Mock()
    client.generate_json.return_value = '{"summary":"A summary","keywords":["faith"],"answerable_questions":["What is faith?"]}'
    settings = Settings(_env_file=None, jina_api_key="key", google_api_key="key")
    metadata = MetadataGenerator(client, settings).generate("source text")
    assert metadata["keywords"] == ["faith"]
