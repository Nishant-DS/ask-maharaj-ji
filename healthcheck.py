"""Operational health check for local configuration and optional integrations."""

from __future__ import annotations

import argparse
import platform
import sys

from app.config import ConfigurationError, Settings
from app.database.db import Database
from app.services.ai.embedding_factory import EmbeddingFactory
from app.services.ai.gemini_client import GeminiClient


def _report(name: str, passed: bool, detail: str = "") -> bool:
    print(f"{'PASS' if passed else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Ask Maharaj Ji ingestion dependencies.")
    parser.add_argument("--database", action="store_true", help="Also check PostgreSQL and pgvector")
    args = parser.parse_args()
    passed = _report("Python version", sys.version_info >= (3, 12), platform.python_version())
    try:
        settings = Settings()
        _ = settings.embedding_dimensions
        config_ok = _report("Configuration loaded", True, "project-root .env")
    except Exception as error:
        _report("Configuration loaded", False, str(error))
        return 1
    passed = passed and config_ok
    try:
        settings.validate_ai_requirements()
        _report("API key configuration", True)
    except ConfigurationError as error:
        passed = _report("API key configuration", False, str(error)) and passed
    if settings.google_api_key:
        try:
            GeminiClient(settings).generate_json(settings.llm_model, 'Return exactly {"ok": true}.')
            _report("Gemini API reachable", True)
        except Exception as error:
            passed = _report("Gemini API reachable", False, str(error)) and passed
    else:
        passed = _report("Gemini API reachable", False, "GOOGLE_API_KEY is missing") and passed
    try:
        EmbeddingFactory.create(settings).embed_batch(["health check"])
        _report("Jina API reachable", True, f"{settings.embedding_dimensions} dimensions")
    except Exception as error:
        passed = _report("Jina API reachable / dimension", False, str(error)) and passed
    if args.database:
        try:
            settings.validate_database_requirements()
            Database(settings).verify(settings.embedding_dimensions)
            _report("PostgreSQL reachable and pgvector", True)
        except Exception as error:
            passed = _report("PostgreSQL reachable and pgvector", False, str(error)) and passed
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
