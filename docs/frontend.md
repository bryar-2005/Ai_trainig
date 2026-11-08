# Sorani Studio web workspace

The `web/` directory is a Next.js App Router application with public marketing, registration, login and reset pages and protected dashboard, project, editor, video, glossary, QA, export, jobs, model and settings routes. It uses React Query, virtualized cue lists, local Inter/Vazirmatn/Noto Naskh font packages, keyboard controls, RTL/LTR direction isolation, and English/Sorani navigation and common workflow copy. Server-provided job messages, model warnings and API error details may still be English. Preview the output and have a qualified reviewer check the translation; confidence scores are uncalibrated estimates, not quality guarantees.

## Run locally

1. Follow [backend.md](backend.md) for Python 3.11, PostgreSQL, Redis, migrations, the API, a Celery worker, and beat. Configure a licensed local or cached model separately; none is shipped.
2. Install Node.js 22 or newer and run:

   ```bash
   cd web
   npm ci
   # The browser always uses /api; only the Next server sees this address.
   export INTERNAL_API_ORIGIN=http://127.0.0.1:8000
   npm run dev
   ```

3. Open `http://localhost:3000`. Register a user; an operator can create an administrator with `python -m api.db.seed`. The administrator can register compatible trusted model assets in **Models**. Select/activate one before starting translation. A project can instead be created as a draft, manually reviewed and exported with no model.

Only upload material you are permitted to use. The browser's temporary upload handoff and unsaved editor drafts are held in the current tab's memory, not local storage; they do not survive a full reload. A pending draft is retained across client-side navigation after a failed save, and a full page exit with unsaved text requests browser confirmation. Browser language and theme preferences are stored locally. Cue saves use revision checks: if another editor changes a cue, reload/review the current server revision rather than overwriting it blindly.

The public sample widget only runs a model actually registered in the signed-in workspace. It accepts at most three short lines and queues the same worker infrastructure as other jobs; no examples are presented as model predictions. Video preview renders authorized uploaded media through the authenticated, range-enabled API. Browser codec support varies; MKV is not reliably playable in every browser. Nudge controls modify timestamps only after an explicit click. ASS exports reject times that cannot be represented exactly at centisecond precision.

## Security and deployment

The browser calls only same-origin `/api/*` URLs. Next's server-side rewrite forwards them to `INTERNAL_API_ORIGIN`, which must be a trusted private API origin (for example `http://api:8000` in a container network). Never expose it as a `NEXT_PUBLIC_*` variable, place credentials in it, or use browser-side `localhost` for a remote preview. The Next proxy verifies the public Origin against the forwarded host/protocol before forwarding a same-origin request and the API verifies its signed, session-bound CSRF header. A trusted reverse proxy **must strip client-supplied `X-Forwarded-*` headers and set the correct host and protocol itself**. Do not put the Next server directly behind an untrusted proxy that lets clients choose those headers. Restrict public access to the web service; keep PostgreSQL, Redis, model directories, API, worker and object storage private. TLS and `SORANI_COOKIE_SECURE=true` are mandatory in production.

Authentication uses HTTP-only access/refresh cookies and a readable, session-bound CSRF cookie; the client attaches it on mutations and coordinates refreshes within a tab. Logging out or changing passwords clears sessions and in-memory caches. Personal API tokens are shown once, are stored server-side only as hashes, expire, and can be revoked; treat them as secrets. Password reset is disabled until SMTP host, sender and public HTTPS web origin are configured. Reset requests return the same success response for known and unknown accounts when mail is configured, links expire and are single-use, and reset revokes existing sessions and API tokens. See [backend.md](backend.md) for the associated settings and security details.

## Tests and development

```bash
cd web
npm run typecheck
npm run lint
npm test
npm run build
npx playwright install chromium
# Start the real backend and its dependencies first.
npm run test:e2e
```

The ordinary Playwright suite registers disposable test accounts and exercises authorized synthetic subtitle upload, edit/approve, glossary, export, settings, login, responsive RTL and a browser-generated synthetic WebM. It requires a reachable, **disposable** API database/storage configured for HTTP development cookies. It does not need a translation model. `PLAYWRIGHT_BASE_URL` selects the site; `PLAYWRIGHT_EXTERNAL_WEB=1` tells the test runner to use an already running frontend. `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH` can select a locally installed Chromium. The optional `E2E_SYNTHETIC_MODEL` test requires a deliberately configured random test-only model and worker; `E2E_EXPECT_QUEUED=1` adds a live SSE/cancellation test with the worker deliberately stopped. Such test-only models are rejected outside `SORANI_ENVIRONMENT=test` and are never evidence of translation quality.

The editor supports arrow-key cue navigation, Enter to edit, Ctrl/⌘+S to save, and Ctrl/⌘+Enter to approve and jump to the next flagged cue. Confirmation dialogs are keyboard accessible. On small screens the navigation drawer traps focus and closes with Escape. Virtualized lists and pagination avoid rendering every cue at once; the current editor page contains up to 100 cues. The video timeline and glossary occurrence count load project cues in pages.
