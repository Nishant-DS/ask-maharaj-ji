"""FastAPI application for read-only semantic transcript retrieval."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from app.config import Settings
from app.database.db import Database
from app.database.repository import TranscriptChunkRepository
from app.models.retrieval import RetrievedChunk, RetrievalQuery
from app.services.ai.embedding_factory import EmbeddingFactory
from app.services.retrieval_service import RetrievalService


class HealthResponse(BaseModel):
    """Minimal readiness response."""

    status: str


def create_app(retrieval_service: RetrievalService | None = None) -> FastAPI:
    """Create the API, validating external dependencies at startup when needed."""
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if retrieval_service is not None:
            app.state.retrieval_service = retrieval_service
        else:
            settings = Settings(metadata_generation_enabled=False)
            settings.validate_ai_requirements()
            settings.validate_database_requirements()
            database = Database(settings)
            database.verify(settings.embedding_dimensions)
            app.state.retrieval_service = RetrievalService(
                TranscriptChunkRepository(database.session_factory),
                EmbeddingFactory.create(settings),
            )
        yield

    app = FastAPI(
        title="Ask Maharaj Ji Retrieval API",
        version="0.2.0",
        description="Read-only semantic retrieval over ingested transcript chunks.",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.post("/v1/retrieve", response_model=list[RetrievedChunk])
    def retrieve(payload: RetrievalQuery, request: Request) -> list[RetrievedChunk]:
        """Return the closest transcript chunks without generating an answer."""
        try:
            service: RetrievalService = request.app.state.retrieval_service
            return service.retrieve(payload)
        except Exception as error:
            raise HTTPException(status_code=503, detail="Retrieval is temporarily unavailable.") from error

    return app


app = create_app()
