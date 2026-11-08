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

Celery beat periodically republishes queued jobs that were committed but not delivered and jobs with expired leases. It also runs private feedback retention cleanup hourly and invokes the optional nightly scheduler at 02:00 UTC. Nightly exports require both `SORANI_NIGHTLY_FEEDBACK_EXPORT_ENABLED=true` and the individual user's setting; both default off. Workers use fenced leases and transactional commits; duplicate deliveries do not apply results twice. Cancellation is cooperative at pipeline/optimizer boundaries, not instantaneous during a model call. Keep the API, worker and beat running; a separate scheduler instance should be elected in a multi-replica deployment. Translation and training should use a suitably sized CPU/GPU host and private mounted storage. Avoid exposing PostgreSQL, Redis, the worker, or local model files publicly.

`GET /health` checks PostgreSQL and Redis, `GET /metrics` exposes Prometheus request counters/latency, and `/docs` serves the OpenAPI explorer. The API logs a structured route template, method, status and duration—not request bodies or subtitle text. Disable the reverse proxy's query/body logging as well. Configure trusted proxy headers narrowly; the Redis limiter keys requests by the client address visible to the API and otherwise may see only the proxy address.

For production set `SORANI_ENVIRONMENT=production`, a private, persistent `SORANI_DATABASE_URL`, a strong `SORANI_SECRET_KEY`, and `SORANI_ALLOWED_ORIGINS` as a JSON array of exact HTTPS frontend origins. Secure cookies and Celery dispatch are mandatory in production. Put TLS and an appropriate trusted reverse proxy in front of the API, and protect `/metrics` operationally. Login uses Argon2id; browser sessions use HTTP-only, SameSite cookies and short-lived JWT access tokens. Refresh tokens rotate against a server-side session and replay revokes that session. For every authenticated mutating request, echo the readable `sorani_csrf` cookie as `X-CSRF-Token`; the token is signed and session-bound. Credentialed cross-origin requests must use an allowed origin. Register/login do not require a prior session.

Local files are stored under `SORANI_STORAGE_ROOT` with private permissions and server-generated keys, outside the public web root and Git. The API checks each multipart file size during streaming, validates allowed subtitle/video extensions and basic video signatures, and serves video only after authorization with byte-range support. The OS temporary multipart spool, staging directory and object store must have enough private disk capacity; configure a proxy body limit as a second boundary. Deleting a project removes its uploaded files and database rows and invalidates **all** retained feedback snapshots belonging to its owner. Previously downloaded copies, backups, and separately trained model weights can retain project information; deletion does not retract downloads or automatically untrain/delete derived models. Establish your own retention and backup policies.

For S3-compatible private object storage set `SORANI_STORAGE_BACKEND=s3`, `SORANI_S3_BUCKET`, `SORANI_S3_REGION` and, for MinIO, `SORANI_S3_ENDPOINT_URL`; provide credentials through the standard AWS credential chain. The bucket must already exist and must not be public. `SORANI_STORAGE_ROOT` is still needed for private temporary staging. No credentials or subtitles should be committed.

## Account recovery and personal tokens

Password reset is disabled by default. To enable it, set `SORANI_SMTP_HOST`, `SORANI_SMTP_PORT`, `SORANI_SMTP_SENDER`, and `SORANI_PUBLIC_WEB_ORIGIN` together; optional `SORANI_SMTP_USERNAME` and `SORANI_SMTP_PASSWORD` must be supplied as a pair. `SORANI_SMTP_STARTTLS=true` is the default and production requires STARTTLS and an HTTPS public origin. `SORANI_RESET_MINUTES` controls expiry (5–120 minutes, default 30). Delivery runs after the generic response; failures are logged without recipient, password or token, so configure operational mail monitoring. Reset tokens are random, stored hashed, single-use, and revoke all existing sessions and personal tokens when used. Do not log reset URLs at the reverse proxy.

Authenticated users may create personal bearer tokens with their current password. Tokens are shown once, stored as hashes, expire in 1–365 days and can be revoked. They inherit the account's permissions; use them only with trusted applications. Password changes and resets revoke them as well as active browser sessions. The web workspace and its same-origin proxy are documented in [frontend.md](frontend.md).

## Workflow and safety

1. As administrator, register a trusted local base or merged model directory within `SORANI_MODEL_ROOT`, or an explicit Hub ID. An adapter also requires its Phase 2 directory and completed training manifest. Local adapter/base identity hashes and direction are checked. Hub loading defaults to cached-only; opt into networking explicitly with `SORANI_LOCAL_FILES_ONLY=false`. Administrator-supplied metrics are labeled unverified. Set an active model per direction or select one on a project. This repository does not create or imply a real model version.
2. Register a user, create a project, and upload an authorized SRT/VTT/ASS/SSA file with an optional MP4/MKV/WebM. `POST /api/subtitles/inspect` parses a bounded authorized upload temporarily for the wizard preview and removes the staging file. Enqueue translation and follow `GET /api/jobs/{id}/stream` using SSE; `Last-Event-ID` resumes persisted progress events, and `?after=<event-id>` resumes a newly created stream after refreshing authentication. Cue IDs, source lineage and original timestamps are retained. The API never shifts timestamps automatically. Editing, glossary changes and deletion are blocked while a project job is active.
3. Review and edit cues with `expected_revision`; competing edits return HTTP 409. Approval requires text for spoken cues. Restore an earlier revision through `/api/cues/{id}/revisions/{revision_id}/restore`; this restores editable text/times but intentionally does **not** restore an old machine model or silently reapprove a cue. Re-translation of a cue that shares a reconstructed sentence is flagged for review; existing human text is not overwritten. Compression is a small deterministic rule set, not a generative AI guarantee.
4. Add/import locked glossary terms and inspect QA. Confidence is an **uncalibrated heuristic**, not a correctness probability. Subtitle exports support SRT, VTT, ASS, JSON and TXT, UTF-8/BOM, bilingual and approved-only options, and include machine-assistance provenance. SRT has an HTML-comment preamble; some players may not support such comments, so check compatibility. ASS has centisecond precision; the API rejects unrepresentable millisecond times instead of silently rounding them.
5. Only approved human edits belonging to the requesting user are exported or used for feedback. Human approval is not language identification: records remain `language_verified=false`, and training requires explicit `allow_unverified_language=true`. A minimum of two films is required for film-disjoint train/validation; with exactly two there is no independent test split. Previous **source** sentences are reconstructed from the full film, including intervening unreviewed source cues; unreviewed targets are never used. Continued training loads the existing adapter as trainable and records dataset/base identity and lineage. Validation chrF++ is not a substitute for a reviewed held-out test set.

Profanity is preserved by default. A project can explicitly enable `soften_profanity` only with user-provided target-language `profanity_replacements` pairs (`source`, `replacement`); there is no invented built-in Sorani lexicon. Matching is literal and boundary-aware, applied to machine text only, and changed cues are flagged for human review. Locked-glossary conflicts remain visible. User edits are never automatically softened.

## Retained feedback snapshots

`POST /api/feedback/export` still returns an immediate owner-scoped JSONL download. For an asynchronous retained snapshot, `POST /api/feedback/exports` queues a Celery job (optionally select `direction`); follow `/api/jobs/{id}` and its SSE stream. A completed job contains an export ID, row count and SHA-256, never dialogue. `GET /api/feedback/exports` lists only your own unexpired snapshots; `GET /api/feedback/exports/{id}/download` streams private JSONL and `DELETE /api/feedback/exports/{id}` deletes it. Administrators cannot download another user's snapshots through these endpoints. Each row records human approval and `language_verified=false`; this is not a quality or licensing guarantee. An empty eligible set completes without creating a file.

Nightly creation is disabled by default. When the operator enables `SORANI_NIGHTLY_FEEDBACK_EXPORT_ENABLED=true`, each account may separately opt in through Settings. Beat schedules at most one export per consenting owner and UTC day and skips owners with no eligible edits or another active export. Turning off the user preference before a queued scheduled job executes cancels it. `SORANI_FEEDBACK_EXPORT_RETENTION_DAYS` defaults to 7 (1–30). Expired snapshots cannot be downloaded; beat removes their objects hourly, retrying failures. It also removes abandoned pending uploads and their private staging directories after lost leases. If beat or storage is unavailable, bytes may remain until cleanup succeeds. Project deletion invalidates all of that owner's retained snapshots, but cannot retract downloaded copies or backups. Manage permissions and retention for model weights and all backup copies separately.

## Tests

```bash
python -m pip install -e '.[dev,api,api-test,train]'
# Optional: disposable PostgreSQL/Redis URLs enable real database and live Celery tests.
# WARNING: SORANI_TEST_DATABASE_URL must name a disposable database; the tests drop its API tables.
export SORANI_TEST_DATABASE_URL='postgresql+asyncpg://sorani:sorani@localhost:5432/sorani_test'
export SORANI_TEST_REDIS_URL='redis://localhost:6379/14'
export SORANI_TEST_BROKER_URL='redis://localhost:6379/15'
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 pytest -q tests/test_api*.py tests/test_account_frontend_support.py tests/test_model_identity_lightweight.py tests/test_feedback_exports.py tests/test_demo_seed.py
```

Without those test URLs, HTTP tests use a temporary SQLite file and an in-memory rate limiter; PostgreSQL-specific locking and live Celery/Redis tests skip. Production never permits SQLite, the in-memory limiter, smoke models, or manual task dispatch. The integration suite checks real random-model inference, retranslation, comparison and continued training, not translation quality. Run `alembic upgrade head`, `alembic check`, `alembic downgrade base`, and `alembic upgrade head` only against a disposable database when checking migrations.
