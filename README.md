# Ask Maharaj Ji — Phase 1 Ingestion

Production-oriented ingestion for the Ask Maharaj Ji RAG system. It reads a timestamped transcript CSV, creates row-preserving semantic chunks, generates English retrieval metadata, creates multilingual embeddings, and stores the result in PostgreSQL with pgvector. It can also download captions for every available video in a YouTube playlist before ingesting them.

Phase 1 intentionally includes no retrieval, search, HTTP API, reranking, answer generation, or chat UI.

## Architecture

`CSV → TranscriptParser → SemanticChunker → MetadataGenerator (Gemini) → EmbeddingProvider → TranscriptChunkRepository → PostgreSQL`

The parser, chunker, metadata generator, provider adapters, and repository are independent components. `IngestionService` only coordinates them. Gemini SDK calls are isolated to `app/services/ai/gemini_client.py`; all embedding calls flow through `BaseEmbeddingProvider`.

## Layout

```
ingest.py                         CLI
ingest_playlist.py                playlist download-and-ingest CLI
app/config.py                     validated environment settings
app/database/                     SQLAlchemy pool and repository
app/models/                       transcript and chunk domain models
app/services/                     parser, chunker, orchestration
app/services/ai/                 provider adapters and Gemini gateway
app/prompts/metadata_prompt.txt  externally maintained metadata prompt
transcripts/                     place source CSVs here (optional)
logs/                            application log destination (optional)
```

## Local setup

Requires Python 3.12, PostgreSQL 16, and pgvector. The project supports standard pip editable installs.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
```

Fill in `.env` with database credentials and API keys. All settings are environment variables, so deployment platforms can omit a local `.env` file.

## Database

Enable pgvector, then create the table supplied in the project requirements:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

This project deliberately does not create migrations or alter your schema. Before ingestion it verifies connectivity, the `vector` extension, the `transcript_chunks.embedding` column, and its vector dimension.

### Vector dimension

The database embedding column must be `VECTOR(1024)` for `jina-embeddings-v3`. The dimension is derived from the model in code and is intentionally not an environment variable. The pipeline validates both returned vectors and the database column before writing.

## Health check

Check Python, root-relative `.env` configuration, Gemini, Jina, and the model's embedding dimension:

```bash
python healthcheck.py
```

Add `--database` to also verify PostgreSQL, pgvector, and the `transcript_chunks.embedding` dimension:

```bash
python healthcheck.py --database
```

## Input and run

The CSV must have `start,end,text` headers. Fields are parsed as seconds and source rows are never split.

```csv
start,end,text
12.08,16.92,"महाराज जी..."
16.92,20.50,"भगवत मार्ग..."
```

```bash
python ingest.py transcripts/example.csv \
  --video-id VIDEO_ID \
  --youtube-url https://www.youtube.com/watch?v=VIDEO_ID \
  --title "Discourse title" \
  --speaker "Maharaj Ji" \
  --language hi \
  --discourse-date 2025-01-31
```

Use `--replace` to explicitly delete an already-ingested video's existing chunks before replacing them. Without it, duplicate ingestion fails safely.

### Dry run

Run parsing, chunking, Gemini metadata generation, and Jina embedding generation without supplying database credentials or video metadata and without creating database writes:

```bash
python ingest.py transcripts/example.csv --dry-run
```

The dry-run summary also prints the first five created chunks with their indices, timestamps, and original transcript text. Jina embeds these chunks; it does not create them.

### Playlist ingestion

`ingest_playlist.py` uses `yt-dlp` to resolve a playlist without downloading video or audio. It fetches captions through `youtube-transcript-api`, writes a `start,end,text` CSV for each video to disk, and invokes the same `IngestionService` as the single-CSV CLI, one video at a time.

```bash
python ingest_playlist.py 'https://www.youtube.com/playlist?list=PLAYLIST_ID' \
  --speaker 'Maharaj Ji' \
  --languages hi en \
  --transcript-dir transcripts/maharaj-ji \
  --replace
```

`--languages` is the preferred-caption-language order; it defaults to `hi en`. Use `--max-videos N` to limit a full run to the first `N` videos in playlist order. `--dry-run` processes only the first two playlist videos (or fewer when `--max-videos` is lower), downloads their transcripts, parses them, generates metadata and embeddings, but makes no database writes. It prints the absolute path of every saved CSV. Each downloaded transcript is retained as `<video-id>.csv`, including when later ingestion fails. A missing or unavailable transcript is reported and does not stop subsequent playlist videos; the command exits with status `2` if any item failed.

### Diagnostic retrieval check

After ingestion, use the standalone diagnostic to verify pgvector similarity ordering. It embeds the query with Jina and prints the nearest chunks; it does not provide answer generation or an API.

```bash
python test_retrieval.py "What does Maharaj Ji say about devotion?" --video-id VIDEO_ID
```

## Chunking and metadata

Chunks target 600 tokens with a 120-token overlap by default. Token counting is Unicode-aware and sentence endings (including `।`) are preferred; chunks are still allowed to exceed the target to preserve a long, indivisible CSV row. Transcript text is never translated before metadata or embeddings. Metadata is JSONB with English `summary`, `keywords`, and `answerable_questions`.

## Logging and failures

The CLI emits timestamped structured log records including transcript name, source-row and chunk counts, metadata, embedding, database insertion, and total elapsed times. Errors include a traceback and the command exits non-zero. Network failures are retried with exponential backoff; invalid configuration and authentication errors fail immediately.

## Environment variables

Required for normal ingestion: `GOOGLE_API_KEY`, `JINA_API_KEY`, `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, and `POSTGRES_PASSWORD`. Dry runs require only the API keys. Optional settings are `CHUNK_SIZE`, `CHUNK_OVERLAP`, `EMBEDDING_BATCH_SIZE`, `REQUEST_TIMEOUT_SECONDS`, `MAX_RETRIES`, and `METADATA_GENERATION_ENABLED`.

## Troubleshooting

- **Missing API key:** set the named value in the project-root `.env` file.
- **Dimension mismatch:** use `jina-embeddings-v3` with a `VECTOR(1024)` database column.
- **Database check fails:** run `CREATE EXTENSION IF NOT EXISTS vector;`, create the supplied `transcript_chunks` table, then use `python healthcheck.py --database`.
- **Duplicate video:** re-run full ingestion with `--replace` after confirming replacement is intended.

## Extending

Add embedding providers by implementing `BaseEmbeddingProvider` and registering it in `EmbeddingFactory`; the orchestrator and repository do not change. To accommodate an evolving source format, modify only `TranscriptParser`. Prompt wording lives in `app/prompts/metadata_prompt.txt`, not code.

## Phase 2

Phase 2 can add vector retrieval, filtering, reranking, and an answer-serving API while reusing this ingestion schema and provider abstraction.

## Retrieval API (Phase 2)

The read-only FastAPI service embeds each query with Jina and returns the closest pgvector chunks. It does not generate answers.

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

```bash
curl -X POST http://127.0.0.1:8000/v1/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"query":"What does Maharaj Ji say about devotion?","limit":5}'
```

Use `GET /health` for readiness. The API requires `JINA_API_KEY` and PostgreSQL settings; it intentionally does not require Gemini metadata generation.

### Browser CORS

The retrieval API only permits origins listed in `CORS_ALLOWED_ORIGINS`. For local React development, keep `http://localhost:5173,http://127.0.0.1:5173`; add your final HTTPS frontend domain when it is deployed. Wildcard origins are rejected.
