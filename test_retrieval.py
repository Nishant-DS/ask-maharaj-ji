"""Run a one-off pgvector similarity check against ingested transcript chunks."""

from __future__ import annotations

import argparse
import logging
import sys

from sqlalchemy import text

from app.config import Settings
from app.database.db import Database
from app.services.ai.embedding_factory import EmbeddingFactory


def parse_args() -> argparse.Namespace:
    """Parse a query and optional video filter for diagnostic retrieval."""
    parser = argparse.ArgumentParser(description="Print the nearest ingested chunks for a query.")
    parser.add_argument("query", help="Question or phrase to embed with Jina")
    parser.add_argument("--video-id", help="Restrict results to one YouTube video ID")
    parser.add_argument("--limit", type=int, default=5, choices=range(1, 21), metavar="1-20")
    return parser.parse_args()


def vector_literal(vector: list[float]) -> str:
    """Format a verified float vector safely for pgvector's text input."""
    return "[" + ",".join(str(value) for value in vector) + "]"


def main() -> int:
    """Embed the query, execute cosine-distance ordering, and print the results."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = parse_args()
    try:
        # Metadata is unnecessary for this embedding-only diagnostic.
        settings = Settings(metadata_generation_enabled=False)
        settings.validate_ai_requirements()
        settings.validate_database_requirements()
        database = Database(settings)
        database.verify(settings.embedding_dimensions)
        embedding = EmbeddingFactory.create(settings).embed_batch([args.query])[0]
        sql = """
            SELECT youtube_video_id, title, chunk_index, start_second, end_second, chunk_text,
                   embedding <=> CAST(:query_vector AS vector) AS cosine_distance
            FROM transcript_chunks
            WHERE embedding IS NOT NULL
        """
        params: dict[str, object] = {"query_vector": vector_literal(embedding), "limit": args.limit}
        if args.video_id:
            sql += " AND youtube_video_id = :video_id"
            params["video_id"] = args.video_id
        sql += " ORDER BY embedding <=> CAST(:query_vector AS vector) LIMIT :limit"
        with database.engine.connect() as connection:
            rows = connection.execute(text(sql), params).mappings().all()
        if not rows:
            print("No embedded chunks found for this filter.")
            return 0
        print(f"Top {len(rows)} chunks for: {args.query!r}\n")
        for rank, row in enumerate(rows, start=1):
            print(
                f"{rank}. {row['title']} | chunk {row['chunk_index']} | "
                f"{row['start_second']}s–{row['end_second']}s | "
                f"distance={row['cosine_distance']:.4f}"
            )
            print(f"   {row['chunk_text']}\n")
        return 0
    except Exception as error:
        logging.getLogger(__name__).exception("Retrieval diagnostic failed: %s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
