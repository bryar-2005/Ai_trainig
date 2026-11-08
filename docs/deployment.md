# Deployment, operations, and privacy

The included Compose configuration runs the Next.js web app, FastAPI, one CPU Celery worker, Celery beat, PostgreSQL, Redis, and a private S3-compatible MinIO instance. The GPU overlay replaces the CPU worker with an NVIDIA worker. No trained model, weights, actual subtitle corpus, benchmark, evaluation score, default administrator, or usable credentials are supplied. Users must have the rights to every subtitle, model, and derived artifact they process. The default NLLB-200 distilled 600M checkpoint is CC-BY-NC 4.0: review its model card and restrictions before use, especially for commercial deployments.

## Prerequisites and important limits

- Docker Engine with the Compose v2 plugin, build access to Python/npm/Go package registries and GitHub, and enough disk/RAM for the Python, Node, PyTorch, and object-storage images. Model weights and training need additional resources. The first build downloads dependencies but **not model weights or subtitle corpora**.
- The included MinIO Community Edition and `mc` are built from pinned upstream source commits (`docker/minio.Dockerfile`); the upstream repositories are archived and no longer maintained. The implementation is useful for a private, single-host deployment, not a substitute for a supported, patched production object-storage service. MinIO is AGPLv3: review your obligations. For production, use a supported S3-compatible service and adapt the common S3 environment variables and bucket setup, or maintain and audit your own source build. Do not publicly expose the bundled MinIO endpoint or console.
- Named volumes persist PostgreSQL, Redis, MinIO objects, private staging, model assets, and the Hugging Face cache. The web service alone publishes a port, bound to `127.0.0.1:3000` by default. API, Redis, PostgreSQL, MinIO, and metrics have no published host port. Only the trusted frontend proxy should reach the API.
- Compose uses dedicated, bucket-limited MinIO application credentials, separate from the root pair. It creates the bucket privately and checks the application account can list it. Credentials, `.env`, Docker access, backups, and volumes require host-level protection. Compose environment variables and `docker compose config` can expose secrets to host administrators; do not paste their output into issue trackers or logs.

## Start the CPU stack

Copy the Compose template and generate unique URL-safe secrets without printing them. The example below refuses to overwrite already-filled secret values:

```bash
cp .env.compose.example .env
chmod 600 .env
python3 - <<'PY'
from pathlib import Path
import secrets
path = Path('.env')
content = path.read_text()
values = {
    'POSTGRES_PASSWORD': secrets.token_urlsafe(32),
    'REDIS_PASSWORD': secrets.token_urlsafe(32),
    'MINIO_ROOT_USER': 'root' + secrets.token_hex(4),
    'MINIO_ROOT_PASSWORD': secrets.token_urlsafe(32),
    'MINIO_APP_ACCESS_KEY': 'app' + secrets.token_hex(4),
    'MINIO_APP_SECRET_KEY': secrets.token_urlsafe(32),
    'SORANI_SECRET_KEY': secrets.token_urlsafe(48),
}
for key, value in values.items():
    old = f'{key}=\n'
    if old not in content:
        raise SystemExit(f'{key} is missing or already set; refusing to overwrite')
    content = content.replace(old, f'{key}={value}\n', 1)
path.write_text(content)
PY

docker compose config --quiet
docker compose up -d --build --wait
```

Use URL-safe passwords because PostgreSQL and Redis credentials are embedded in internal connection URLs. Keep the signing key stable: changing it invalidates all sessions. Changing a PostgreSQL/Redis/MinIO password in `.env` alone does not safely rotate existing persistent credentials. For the MinIO application account, update the user in MinIO and `.env` together; the initializer fails closed if the saved account and supplied secret do not match. Manage rotation with a documented maintenance procedure and backups.

Visit `http://localhost:3000` on the host. These development defaults are **not** safe for an Internet-facing site. The one-shot `migrate` service runs Alembic before the API, worker and beat start; inspect `docker compose ps` or `docker compose logs api worker-cpu beat` if startup fails. The application can create/edit/review/export draft subtitles with no model. An operator can explicitly create an administrator without a default password:

```bash
docker compose exec api python -m api.db.seed --email admin@example.com --display-name Administrator
```

The command prompts twice without echoing the password and refuses to promote/overwrite an existing account. For a secret manager use `--password-stdin` with a protected standard-input stream, not a password in arguments or shell history.

### Optional synthetic demo

Only in development/test, an operator may deliberately create a **new ordinary user** and one original, three-cue, untranslated synthetic project:

```bash
docker compose exec api python -m api.db.seed_demo --create-synthetic-demo \
  --email demo@example.com --display-name 'Demo reviewer'
```

The password is prompted, no existing user is changed, and the command refuses to run in production. The demo installs no model, translations, metrics, licensed subtitle corpus, or training data. It is not a quality benchmark. Register your own real model assets separately and delete the synthetic project/account when no longer needed.

## Add authorized model assets

Obtain a compatible base/merged model or adapter yourself after reviewing licenses, security and model-card limitations. Never accept untrusted custom model code. Offline loading is the default (`SORANI_LOCAL_FILES_ONLY=true`). For a CPU deployment, one way to place a local directory into the persistent model volume is:

```bash
# Replace the source with a directory you are licensed and authorized to use.
docker compose exec -u root worker-cpu mkdir -p /app/outputs/models/my-base
docker compose cp /path/to/authorized/model/. worker-cpu:/app/outputs/models/my-base/
docker compose exec -u root worker-cpu chown -R 10001:10001 /app/outputs/models/my-base
```

For a GPU deployment use `worker-gpu` in those commands. The API mounts this volume read-only; the worker needs read/write access to store feedback-trained adapters. As an administrator, register `/app/outputs/models/my-base` in **Models** for the correct direction and activate/select it. For an adapter, provide its matching base and the complete Phase 2 adapter directory with its training manifest; identity hashes and direction are checked. For a Hub ID, only a trusted, explicitly registered model can be loaded, and `SORANI_LOCAL_FILES_ONLY=false` is required for network access. Do not treat admin-supplied metrics, confidence, or unverified language identification as measured quality. See [backend.md](backend.md) and the training and evaluation sections of the [README](../README.md).

## Optional NVIDIA GPU worker

Install a compatible NVIDIA driver and NVIDIA Container Toolkit on the host. The overlay builds a separate PyTorch 2.6 CUDA 12.4 image and requests NVIDIA GPUs through Compose. It is not an automatic CPU fallback; if CUDA or its driver is unavailable, jobs will fail. Stop the CPU stack first when switching so both workers do not compete for the same queue:

```bash
docker compose down                         # retains named volumes
# Reuses the same .env and named model/storage volumes.
docker compose -f compose.yaml -f compose.gpu.yaml --profile gpu up -d --build --wait
docker compose -f compose.yaml -f compose.gpu.yaml --profile gpu exec worker-gpu \
  python -c 'import torch; print(torch.cuda.is_available())'
```

The command must print `True` before GPU jobs are expected to work. The overlay puts `worker-cpu` behind the inactive `cpu` profile; do not enable both profiles unless that competition is intentional. To return to CPU, bring the overlay down **without `--volumes`** and run the base Compose command again. Tune memory, batching and worker count for your own model and device; no GPU benchmark is claimed.

## Production boundary and TLS

Terminate HTTPS in a maintained reverse proxy in front of the loopback-bound web service. Set `SORANI_ENVIRONMENT=production`, `SORANI_COOKIE_SECURE=true`, and `SORANI_ALLOWED_ORIGINS` to an exact JSON array of trusted HTTPS browser origins, e.g. `["https://studio.example.com"]`. Recreate the services after changing settings. Password-reset SMTP is disabled until host, sender and public origin are set together; production requires HTTPS and SMTP STARTTLS. Use a secret manager and restricted host access rather than sharing `.env`.

The Next.js server forwards same-origin `/api/*` requests to `http://api:8000` on the private Compose network. Its rewrite is compiled at build time: if `INTERNAL_API_ORIGIN` changes, set it consistently as a web build argument and runtime variable, then rebuild. Browser code must not call a remote service at `localhost`. A trusted proxy **must overwrite**, not trust, incoming `X-Forwarded-Host` and `X-Forwarded-Proto`; configure a fixed allowed public hostname, strip client-supplied forwarding headers, and prevent direct access around the proxy. Forward the correct HTTPS scheme/host so the Next origin check succeeds. Disable request-body, cookie, authorization, reset-token, and sensitive query logging. Configure an upload limit at least as large as your permitted video plus subtitle and multipart overhead, timeouts suitable for uploads, and no proxy buffering for SSE job streams. Restrict `/metrics` and infrastructure access. Tune rate limits for the deployment: the backend sees the client address presented by the proxy, which may otherwise group users together.

Backups and model downloads need the same confidentiality and authorization controls as original uploads. Use a supported object store, encryption at rest, network restrictions, secrets management, patching, monitoring, and your own threat model; this single-host Compose setup is not a high-availability production cluster.

## Feedback, retention, and deletion

In **Models & feedback**, a user can immediately download approved edits or start a private asynchronous snapshot job. The retained snapshot panel lists only that user's exports and allows download/deletion. JSONL records contain approved human edits with `language_verified=false`, direction, film/cue IDs and revision/provenance. Existing unapproved targets are excluded. Creating a snapshot does not train a model or make a quality claim.

Nightly snapshots require **both** the operator setting `SORANI_NIGHTLY_FEEDBACK_EXPORT_ENABLED=true` and each user's explicit checkbox in Settings. The default for both is off. Beat checks at 02:00 UTC and queues at most one job per opted-in account per UTC day, only when eligible feedback exists; a concurrent active snapshot is not duplicated. Revoking consent before a queued scheduled job runs cancels it. Manual export remains available without the nightly setting. Results are private; owners can delete them at any time. `SORANI_FEEDBACK_EXPORT_RETENTION_DAYS` defaults to 7 and accepts 1–30. Expired files are immediately inaccessible and an hourly beat task removes them and abandoned pending uploads. If beat/storage is unavailable, physical cleanup waits until it recovers. Private staging may also require operator cleanup after abrupt host termination; do not remove active workers' scratch data.

Deleting a project invalidates **all** retained snapshots of that project owner, not just the snapshots known to include the project. Downloads already copied elsewhere, backups, logs, and model weights trained from feedback are not automatically retracted, deleted, or untrained. Establish and honor retention/deletion policies for every derived copy and model.

## Maintenance and recovery

- `docker compose ps`, `docker compose logs api worker-cpu beat`, and the internal `/health` check expose service state. API logs do not contain request bodies, but protect all logs and configure the proxy accordingly. Keep exactly one beat scheduler active in a multi-replica deployment; the database outbox and fenced leases recover undelivered and stale jobs.
- Before an upgrade, back up PostgreSQL, the private S3 bucket, model assets, configuration/secrets and any relevant caches together. Quiesce writers or use a coordinated database/object-store backup so references match. Store backups encrypted, access-controlled, and subject to the same deletion/retention policy. Test restoration in an isolated environment. The MinIO named volume is an internal storage format; use a supported MinIO/S3 backup method or a consistent stopped-service volume snapshot, not a public bucket.
- `docker compose down` stops and removes containers but keeps named volumes. **`docker compose down --volumes` irreversibly removes the named volumes and their data.** Keep independent backups before any destructive command.
- Migrations run on startup. Downgrades may refuse while newer job/export records exist; never run downgrade/reset commands against real data without an explicit tested rollback plan. After a worker crash, beat retries stale leases; cancellation is cooperative and can wait for a model call. Storage errors can leave pending cleanup; restore connectivity and monitor beat.
- To validate a disposable deployment end-to-end, run `python scripts/compose_smoke.py` after startup. It creates a synthetic account/project, uploads an original one-line subtitle, approves an edit, queues a real Celery snapshot, downloads it, then deletes the project; the account remains and should be removed according to the operator's test-data policy. Never run it against a shared production instance without permission. The CI workflow also validates both Compose files and exercises the CPU stack.
