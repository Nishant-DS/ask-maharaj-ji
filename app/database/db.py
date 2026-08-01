"""SQLAlchemy engine and startup database checks."""

from __future__ import annotations

from collections.abc import Callable

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings


class Database:
    """Owns the connection pool and verifies required PostgreSQL capabilities."""

    def __init__(self, settings: Settings) -> None:
        self.engine: Engine = create_engine(
            settings.database_url, pool_pre_ping=True, pool_size=5, max_overflow=10
        )
        self.session_factory: Callable[[], Session] = sessionmaker(
            bind=self.engine, expire_on_commit=False
        )

    def verify(self, expected_dimensions: int) -> None:
        """Fail early when PostgreSQL, pgvector, or its schema are not ready."""
        with self.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            extension = connection.execute(
                text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            ).scalar()
            if extension is None:
                raise RuntimeError("pgvector extension is not installed in this database")
            dimension = connection.execute(
                text("SELECT atttypmod FROM pg_attribute "
                     "WHERE attrelid = 'transcript_chunks'::regclass "
                     "AND attname = 'embedding' AND NOT attisdropped")
            ).scalar_one_or_none()
            if dimension is None:
                raise RuntimeError("transcript_chunks.embedding column was not found")
            # pgvector stores vector(n) dimensions directly in atttypmod.
            actual_dimensions = int(dimension)
            if actual_dimensions != expected_dimensions:
                raise RuntimeError(
                    f"Embedding dimension mismatch: database VECTOR({actual_dimensions}), "
                    f"configured model expects {expected_dimensions}. Alter the column or select a matching model."
                )
