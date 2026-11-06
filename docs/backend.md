# Backend API and worker

The API and workers run real inference and continued LoRA training; they do not ship pretrained weights, a corpus, a reviewed benchmark, or a claim of translation quality. Only upload subtitles and video that you are authorized to process. All examples and smoke-model tests use invented text and a random, **non-translation** model.

## Requirements and startup

Install Python 3.11, PostgreSQL 16, Redis 7, and a licensed NLLB-compatible model or a completed Phase 2 adapter. In the repository root:

```bash
python -m pip install -e '.[api,train]'
cp .env.example .env
# Edit .env for your own PostgreSQL, Redis, allowed browser origins, and model directory.
# Generate a long, random signing key and put it in your environment or a secret store:
export SORANI_SECRET_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
python -m alembic upgrade head
python -m api.db.seed --email admin@example.com --display-name Administrator
```

The last command interactively reads and confirms the initial administrator's password. For a secret manager, `--password-stdin` reads one line from standard input without putting it in command-line arguments or shell history. It does not promote an existing account or insert fake model versions. Keep the signing key stable across restarts; replacing it invalidates sessions. `.env.example` contains **local development** connection defaults, not production credentials.

Start these three processes, with the same settings and storage mounts:

```bash
uvicorn api.main:create_app --factory --host 0.0.0.0 --port 8000 --no-access-log
celery -A api.worker:celery_app worker --loglevel=INFO --concurrency=1
celery -A api.worker:celery_app beat --loglevel=INFO
```

Celery beat periodically republishes queued jobs that were committed but not delivered and jobs with expired leases. Workers use fenced leases and transactional commits; duplicate deliveries do not apply results twice. Cancellation is cooperative at pipeline/optimizer boundaries, not instantaneous during a model call. Keep the API, worker and beat running; a separate scheduler instance should be elected in a multi-replica deployment. Translation and training should use a suitably sized CPU/GPU host and private mounted storage. Avoid exposing PostgreSQL, Redis, the worker, or local model files publicly.

`GET /health` checks PostgreSQL and Redis, `GET /metrics` exposes Prometheus request counters/latency, and `/docs` serves the OpenAPI explorer. The API logs a structured route template, method, status and duration—not request bodies or subtitle text. Disable the reverse proxy's query/body logging as well. Configure trusted proxy headers narrowly; the Redis limiter keys requests by the client address visible to the API and otherwise may see only the proxy address.

For production set `SORANI_ENVIRONMENT=production`, a private, persistent `SORANI_DATABASE_URL`, a strong `SORANI_SECRET_KEY`, and `SORANI_ALLOWED_ORIGINS` as a JSON array of exact HTTPS frontend origins. Secure cookies and Celery dispatch are mandatory in production. Put TLS and an appropriate trusted reverse proxy in front of the API, and protect `/metrics` operationally. Login uses Argon2id; browser sessions use HTTP-only, SameSite cookies and short-lived JWT access tokens. Refresh tokens rotate against a server-side session and replay revokes that session. For every authenticated mutating request, echo the readable `sorani_csrf` cookie as `X-CSRF-Token`; the token is signed and session-bound. Credentialed cross-origin requests must use an allowed origin. Register/login do not require a prior session.

Local files are stored under `SORANI_STORAGE_ROOT` with private permissions and server-generated keys, outside the public web root and Git. The API checks each multipart file size during streaming, validates allowed subtitle/video extensions and basic video signatures, and serves video only after authorization with byte-range support. The OS temporary multipart spool, staging directory and object store must have enough private disk capacity; configure a proxy body limit as a second boundary. Deleting a project removes its uploaded files and database rows. Separately trained model weights can encode information from feedback; project deletion does not automatically untrain or delete those derived models. Establish your own retention and backup policies.

For S3-compatible private object storage set `SORANI_STORAGE_BACKEND=s3`, `SORANI_S3_BUCKET`, `SORANI_S3_REGION` and, for MinIO, `SORANI_S3_ENDPOINT_URL`; provide credentials through the standard AWS credential chain. The bucket must already exist and must not be public. `SORANI_STORAGE_ROOT` is still needed for private temporary staging. No credentials or subtitles should be committed.

## Workflow and safety

1. As administrator, register a trusted local base or merged model directory within `SORANI_MODEL_ROOT`, or an explicit Hub ID. An adapter also requires its Phase 2 directory and completed training manifest. Local adapter/base identity hashes and direction are checked. Hub loading defaults to cached-only; opt into networking explicitly with `SORANI_LOCAL_FILES_ONLY=false`. Administrator-supplied metrics are labeled unverified. Set an active model per direction or select one on a project. This repository does not create or imply a real model version.
2. Register a user, create a project, and upload an authorized SRT/VTT/ASS/SSA file with an optional MP4/MKV/WebM. Enqueue translation and follow `GET /api/jobs/{id}/stream` using SSE; `Last-Event-ID` resumes persisted progress events. Cue IDs, source lineage and original timestamps are retained. The API never shifts timestamps automatically. Editing, glossary changes and deletion are blocked while a project job is active.
3. Review and edit cues with `expected_revision`; competing edits return HTTP 409. Approval requires text for spoken cues. Restore an earlier revision through `/api/cues/{id}/revisions/{revision_id}/restore`; this restores editable text/times but intentionally does **not** restore an old machine model or silently reapprove a cue. Re-translation of a cue that shares a reconstructed sentence is flagged for review; existing human text is not overwritten. Compression is a small deterministic rule set, not a generative AI guarantee.
4. Add/import locked glossary terms and inspect QA. Confidence is an **uncalibrated heuristic**, not a correctness probability. Subtitle exports support SRT, VTT, ASS, JSON and TXT, UTF-8/BOM, bilingual and approved-only options, and include machine-assistance provenance. SRT has an HTML-comment preamble; some players may not support such comments, so check compatibility. ASS has centisecond precision; the API rejects unrepresentable millisecond times instead of silently rounding them.
5. Only approved human edits belonging to the requesting user are exported or used for feedback. Human approval is not language identification: records remain `language_verified=false`, and training requires explicit `allow_unverified_language=true`. A minimum of two films is required for film-disjoint train/validation; with exactly two there is no independent test split. Previous **source** sentences are reconstructed from the full film, including intervening unreviewed source cues; unreviewed targets are never used. Continued training loads the existing adapter as trainable and records dataset/base identity and lineage. Validation chrF++ is not a substitute for a reviewed held-out test set.

Profanity is preserved by default. A project can explicitly enable `soften_profanity` only with user-provided target-language `profanity_replacements` pairs (`source`, `replacement`); there is no invented built-in Sorani lexicon. Matching is literal and boundary-aware, applied to machine text only, and changed cues are flagged for human review. Locked-glossary conflicts remain visible. User edits are never automatically softened.

## Tests

```bash
python -m pip install -e '.[dev,api,api-test,train]'
# Optional: disposable PostgreSQL/Redis URLs enable real database and live Celery tests.
# WARNING: SORANI_TEST_DATABASE_URL must name a disposable database; the tests drop its API tables.
export SORANI_TEST_DATABASE_URL='postgresql+asyncpg://sorani:sorani@localhost:5432/sorani_test'
export SORANI_TEST_REDIS_URL='redis://localhost:6379/14'
export SORANI_TEST_BROKER_URL='redis://localhost:6379/15'
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 pytest -q tests/test_api*.py
```

Without those test URLs, HTTP tests use a temporary SQLite file and an in-memory rate limiter; PostgreSQL-specific locking and live Celery/Redis tests skip. Production never permits SQLite, the in-memory limiter, smoke models, or manual task dispatch. The integration suite checks real random-model inference, retranslation, comparison and continued training, not translation quality. Run `alembic upgrade head`, `alembic check`, `alembic downgrade base`, and `alembic upgrade head` only against a disposable database when checking migrations.
