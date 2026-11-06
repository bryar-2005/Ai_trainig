# Requested specification

This is the user's requested specification, preserved as a project reference. Work proceeds in phases. The active Arena checkout and session branch take precedence over requests below to reinitialize the repository or push to `main`.

# ROLE & CORE DIRECTIVES
You are a principal ML engineer and full-stack architect. Build a complete, production-grade,
multi-page web application and end-to-end AI training/inference pipeline for English ↔ Kurdish
Sorani (Central Kurdish, `ckb`, Arabic script) film and TV subtitle translation.

CRITICAL RULES:
1. **No Placeholders**: Write complete, runnable, production-ready code. NEVER leave `# TODO`,
   `pass`, `...`, mock data stubs, or summarized components.
2. **True Multi-Page Architecture**: The frontend MUST be a real multi-page application using
   Next.js 14+ App Router with separate URL routes, nested `layout.tsx` shells, `loading.tsx`
   skeletons, `error.tsx` boundaries, and `middleware.ts` auth guards. NEVER collapse distinct
   pages into tabs on a single page.
3. **GitHub Repository (`AI_training`)**: Create and manage this entire project inside a Git
   repository named `AI_training` and push all code, configurations, migrations, tests, and
   documentation to the user's GitHub repository `AI_training`.
4. **Explicit Assumptions**: If any minor requirement is ambiguous, state your architectural
   assumption explicitly and implement the complete solution.

# GITHUB REPOSITORY & VERSION CONTROL WORKFLOW
At the start of Phase 0 and after every phase:
1. Initialize the project directory as `AI_training` (`git init -b main`).
2. Create a strict `.gitignore` that excludes secrets (`.env`), large raw datasets, model
   checkpoints/weights (`*.safetensors`, `*.bin`, `*.pt`), video files, `node_modules/`, `.next/`,
   `__pycache__/`, and `.venv/`.
3. Check GitHub CLI authentication (`gh auth status`):
   - If authenticated and `AI_training` does not yet exist on GitHub, create it via
     `gh repo create AI_training --private --source=. --remote=origin` (or `--public` if
     configured).
   - If `origin` already exists or `gh` is unauthenticated, configure `origin` for
     `AI_training` and provide exact push instructions if credentials are required.
4. Commit after every completed & verified phase using clean Conventional Commits
   (`feat(data): ...`, `feat(train): ...`, `feat(web): ...`) and run `git push -u origin main`.

# PRODUCT GOAL & OVERVIEW
A user signs up, uploads English or Kurdish Sorani subtitle files (`.srt`, `.vtt`, `.ass`/`.ssa`)
and optional video files (`.mp4`, `.mkv`, `.webm`), configures translation settings, monitors
background AI jobs in real time via SSE, reviews/edits translations in a dedicated studio,
previews subtitles synced over video, manages character glossaries, runs automated QA audits,
and exports broadcast-ready subtitle files. Human edits are stored as a structured feedback
corpus to continuously fine-tune and benchmark new AI model versions directly from the web app.

# NON-NEGOTIABLE SUBTITLE & LINGUISTIC REQUIREMENTS
1. **Context-Aware Translation**: Never translate a subtitle cue in isolation. Feed a sliding
   window of the previous 3–5 sentences plus speaker labels.
2. **Sentence Reconstruction**: Merge consecutive cues split mid-sentence, translate the full
   sentence, and re-split across the ORIGINAL timestamps at syntactic boundaries. Timestamps
   must NEVER shift automatically.
3. **Reading Speed & Line Limits**:
   - Target ≤ 17 characters/sec (CPS); hard cap at 21 CPS.
   - Max 42 characters per line, max 2 lines per cue.
   - If a translated cue exceeds the CPS cap, run an automated compression pass (drop filler
     words, use concise Sorani phrasing, remove redundant pronouns) without losing core meaning.
4. **Sorani Orthography Normalization**:
   - Enforce standard Kurdish Arabic-script Unicode codepoints:
     - `ک` (U+06A9) instead of Arabic `ك` (U+0643)
     - `ی` (U+06CC) instead of Arabic `ي` (U+064A) or `ى` (U+0649)
     - Word-final vowel `ە` (U+06D5) vs consonant `ھ` (U+06BE) vs Arabic `ه` (U+0647) / `ة` (U+0629)
     - Preserve Kurdish letters: `ڕ` (U+0695), `ڵ` (U+06B5), `ێ` (U+06CE), `ۆ` (U+06C6),
       `ڤ` (U+06A4), `پ` (U+067E), `چ` (U+0686), `ژ` (U+0698), `گ` (U+06AF)
   - Normalize quotes, dashes, and ellipses; use RTL punctuation: `؟` (U+061F), `،` (U+060C), `؛` (U+061B).
5. **Bidirectional (Bidi) & RTL Integrity**:
   - Wrap Sorani text in `dir="rtl"` with proper Arabic-script shaping and right alignment.
   - Isolate embedded Latin names, numbers, and timestamps with Unicode directional isolates
     (`\u2066...\u2069`) in exported files and `<bdi dir="ltr">...</bdi>` in HTML.
6. **Terminology & Register Consistency**:
   - Extract character names, locations, and recurring terms; enforce identical Sorani
     transliteration/translation across the entire film.
   - Maintain consistent formality (`تۆ` vs `ئێوە`) per speaker relationship.
7. **Full Traceability**:
   - Every output cue must store `source_cue_ids`, `machine_text`, `human_text`,
     `model_version_id`, `confidence_score` (0–100), `cps`, and revision history.

# TECH STACK
- **ML & NLP**: Python 3.11, PyTorch 2.x, Hugging Face `transformers`, `datasets`, `peft` (LoRA),
  `accelerate`, `sentencepiece`, `sacrebleu`, `unbabel-comet`, `datasketch` (MinHash), `pysubs2`,
  `fasttext` / `lingua-language-detector`.
- **Base Model**: `facebook/nllb-200-distilled-600M` (`eng_Latn` ↔ `ckb_Arab`), wrapped behind an
  abstract `BaseTranslator` adapter interface so `NLLB-1.3B` or an instruct LLM can be swapped
  in via config without changing pipeline or API code.
- **Backend API**: FastAPI, Pydantic v2, Uvicorn, SQLAlchemy 2.0 (async), Alembic, PostgreSQL 16.
- **Background Jobs & Realtime**: Celery + Redis for async film translation & LoRA training jobs;
  Server-Sent Events (SSE) for real-time job progress and log streaming.
- **Auth & Security**: JWT (access + refresh tokens in HttpOnly cookies), Argon2 password hashing,
  RBAC (`admin`, `reviewer`, `user`), rate limiting, Prometheus `/metrics`, structured JSON
  logging, health checks, graceful shutdown.
- **Storage**: `StorageBackend` abstraction supporting local disk (dev) and S3/MinIO (prod).
- **Frontend (Multi-Page)**: Next.js 14+ (App Router), TypeScript, Tailwind CSS, `shadcn/ui`,
  Lucide icons, TanStack Query v5, Zustand (editor state), `@tanstack/react-virtual` (virtualized
  subtitle table).
- **Typography**: `Vazirmatn` and `Noto Naskh Arabic` for Sorani RTL text (verified Kurdish glyph
  coverage); `Inter` + `JetBrains Mono` for UI and timestamps.
- **Infrastructure & Testing**: Docker, `docker-compose.yml` (web, api, worker-cpu, postgres,
  redis, minio), `docker-compose.gpu.yml` (NVIDIA GPU worker override), `pytest` (backend/ML),
  `vitest` + Playwright (frontend E2E).

# REPOSITORY STRUCTURE (`AI_training/`)
AI_training/
├── .github/workflows/         # CI: lint, pytest, vitest, baseline-regression check
├── ml/
│   ├── data/
│   │   ├── adapters/          # Local directory subtitle readers (no unauthorized scraping)
│   │   ├── loaders.py         # Parse .srt, .vtt, .ass/.ssa into unified Cue dataclass
│   │   ├── normalize.py       # Sorani Unicode normalization, tag stripping, SFX/speaker parsing
│   │   ├── align.py           # Temporal IoU cue alignment (1:1, 1:n, n:1 merges, IoU ≥ 0.5)
│   │   ├── filter_dedupe.py   # Script ratio, length ratio, lang ID, MinHash LSH (0.90)
│   │   └── build_dataset.py   # Film-level train/val/test split + gold_test.jsonl + stats report
│   ├── train/
│   │   ├── config.yaml
│   │   ├── dataset.py         # Context-window formatting (<ctx>) + provenance sample weighting
│   │   ├── train_lora.py      # LoRA fine-tuning, bf16, grad checkpointing, resume, --smoke-test
│   │   └── merge_adapter.py
│   ├── eval/
│   │   ├── metrics.py         # chrF++, spBLEU, COMET, CPS violation %, glossary adherence %
│   │   ├── compare.py         # 100-line side-by-side HTML diff & A/B model comparison
│   │   └── evaluate.py
│   └── inference/
│       ├── base.py            # Abstract BaseTranslator interface
│       ├── nllb_translator.py # Batched NLLB + LoRA inference with log-prob confidence
│       ├── segmenter.py       # Cue → full sentence → syntactic re-split across timestamps
│       ├── glossary.py        # NER extraction, transliteration proposal, term enforcement
│       ├── compressor.py      # Reading-speed shortening pass for high-CPS cues
│       ├── qa.py              # Full cue & script QA checker + confidence scorer (0–100)
│       └── pipeline.py        # End-to-end 12-step translation orchestrator
├── api/
│   ├── main.py
│   ├── core/                  # config, security, logging, rate_limit, metrics, storage
│   ├── db/                    # base, session, migrations/ (Alembic), seed.py
│   ├── models/                # SQLAlchemy models
│   ├── schemas/               # Pydantic v2 schemas
│   ├── routers/               # auth, dashboard, projects, cues, glossary, jobs, models, export
│   ├── services/              # Business logic
│   └── workers/               # Celery app + translation_tasks.py + training_tasks.py
├── web/
│   ├── middleware.ts          # Auth route protection & redirects
│   ├── app/
│   │   ├── layout.tsx         # Root layout, fonts, theme provider, query provider
│   │   ├── not-found.tsx
│   │   ├── (marketing)/
│   │   │   └── page.tsx       # Public landing page + interactive mini demo
│   │   ├── (auth)/
│   │   │   ├── layout.tsx     # Centered auth card layout
│   │   │   ├── login/page.tsx
│   │   │   ├── register/page.tsx
│   │   │   └── forgot-password/page.tsx
│   │   └── (app)/
│   │       ├── layout.tsx     # Authenticated App Shell (Collapsible Sidebar + Topbar + SSE)
│   │       ├── loading.tsx
│   │       ├── error.tsx
│   │       ├── dashboard/page.tsx
│   │       ├── projects/
│   │       │   ├── page.tsx               # Projects directory (grid/table, search, filters)
│   │       │   ├── new/page.tsx           # 4-step upload & translation config wizard
│   │       │   └── [id]/
│   │       │       ├── layout.tsx         # Project context header & sub-navigation bar
│   │       │       ├── page.tsx           # Project overview, analytics & pipeline summary
│   │       │       ├── editor/page.tsx    # Full-screen synchronized subtitle editor
│   │       │       ├── preview/page.tsx   # Video player + timeline scrubber + RTL overlay
│   │       │       ├── glossary/page.tsx  # Project terminology & character names manager
│   │       │       ├── qa/page.tsx        # QA audit report & one-click auto-fixes
│   │       │       └── export/page.tsx    # Multi-format export & encoding configurator
│   │       ├── jobs/
│   │       │   ├── page.tsx               # All background translation & training jobs
│   │       │   └── [jobId]/page.tsx       # Live job progress, pipeline stages & SSE logs
│   │       ├── models/page.tsx            # AI model registry, benchmarks, A/B test & retraining
│   │       └── settings/page.tsx          # User profile, subtitle defaults, API keys
│   ├── components/            # Shared UI, RTL helpers, editor components, video overlay
│   └── lib/                   # Typed API client, auth helpers, RTL/CPS utilities
├── tests/                     # pytest (ml + api) & Playwright E2E
├── docker-compose.yml
├── docker-compose.gpu.yml
├── .env.example
├── .gitignore
└── README.md

---

# PHASE 0 — REPOSITORY INITIALIZATION & GITHUB SETUP
1. Create the `AI_training` workspace, initialize Git (`main` branch), add `.gitignore`,
   `.env.example`, and save this specification as `SPEC.md`.
2. Connect/create the GitHub repository `AI_training` and push the initial commit.

# PHASE 1 — DATA PREPARATION & ALIGNMENT (`ml/data/`)
Build a CLI-runnable pipeline that reads local parallel subtitle directories (no unauthorized
scraping) and produces a clean Hugging Face `DatasetDict` + `ml/eval/gold_test.jsonl`:
1. Parse `.srt`, `.vtt`, `.ass`/`.ssa` into `Cue(index, start_ms, end_ms, text, speaker, style)`.
2. Strip HTML/ASS formatting tags, ads/watermark lines, and music symbols; extract speaker
   prefixes (`JOHN:`) and sound effects (`[DOOR SLAMS]`) into structured fields.
3. Align English ↔ Sorani cues by temporal IoU (reject pairs with IoU < 0.5, tolerance ±500ms),
   supporting 1:1, 1:n, and n:1 cue merges.
4. Filter bad pairs: empty text, identical source/target, length ratio outside `[0.4, 2.5]`,
   Sorani side < 80% Arabic-script codepoints, and language ID verification (`fasttext`/`lingua`).
5. Apply Sorani Unicode normalization + quote/dash/ellipsis normalization.
6. Deduplicate near-identical pairs using MinHash LSH at 0.90 similarity.
7. Tag each pair with `provenance` (`human` | `machine` | `unknown`) and `quality_score` (0–1).
8. Split into `train` / `val` / `test` **grouped by film ID** (never by line, preventing leakage).
   Export 1,000 stratified pairs to `ml/eval/gold_test.jsonl` for human verification.
9. Generate a JSON + Markdown dataset stats report: pair count, film counts, token length
   distributions, provenance mix, per-film counts, and OOV rate.

# PHASE 2 — MODEL TRAINING (`ml/train/`)
Implement `train_lora.py` driven by `config.yaml`:
- Base model: `facebook/nllb-200-distilled-600M` (`eng_Latn` ↔ `ckb_Arab`).
- LoRA config: `r=16`, `lora_alpha=32`, `lora_dropout=0.05`,
  `target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]`.
- Context-aware training format: prepend up to 3 previous source sentences separated by `<ctx>`
  sentinel token and compute loss only on the final target sentence.
- Provenance-weighted loss: weight `human` 1.5x, `unknown` 1.0x, `machine` 0.5x.
- `bf16`, gradient checkpointing, gradient accumulation, cosine LR `2e-4`, warmup `500`, `3` epochs,
  max source/target length `256`, early stopping on validation `chrF++`, W&B/TensorBoard logging,
  checkpoint every N steps, and full checkpoint resumption.
- Include `--smoke-test` mode (`--limit 200` to `500` samples) that completes in < 5 minutes on
  CPU/single GPU to verify end-to-end training, plus `merge_adapter.py`.

# PHASE 3 — EVALUATION (`ml/eval/`)
Compare Zero-Shot Baseline vs. Fine-Tuned Adapter on `gold_test.jsonl`:
- Metrics: `chrF++` (primary — best for morphologically rich Kurdish), `spBLEU`, `COMET`,
  reading-speed CPS violation rate, glossary adherence rate, and length ratio distribution.
- Generate a 100-line side-by-side HTML comparison report with highlighted diffs for human review.
- Exit with a non-zero status code if fine-tuned `chrF++` degrades below baseline.

# PHASE 4 — 12-STEP TRANSLATION & POST-PROCESSING ENGINE (`ml/inference/`)
Implement the 12-step translation pipeline in `pipeline.py` with progress callbacks:
1. Parse input subtitle file into cues.
2. Extract speaker labels and SFX cues.
3. **Sentence Reconstruction**: Merge consecutive cues where cue *i* does not end in terminal
   punctuation (`. ? ! ؟`) or ends with a continuation hyphen/ellipsis; record exact character
   offsets of each original cue inside the merged sentence.
4. **Glossary Pass 1 (NER)**: Extract person/place/organization names across the full English
   script; propose Sorani transliterations and merge with user-locked glossary terms.
5. **Contextual Batched Translation**: Translate reconstructed sentences with sliding context
   (previous 3 sentences) + relevant glossary hints; record token log-probabilities.
6. **Glossary Enforcement**: Verify locked terms appear correctly in the Sorani output; retry with
   stronger constraint or deterministic replacement if missing.
7. **Syntactic Re-Split**: Distribute translated Sorani text back across the original cue
   timestamps proportional to source character offsets, snapping strictly to whitespace and
   syntactic boundaries (conjunctions, prepositions, clause breaks — never mid-word).
8. **Reading-Speed Compression**: For any cue exceeding target CPS (default 17, hard cap 21), run
   an automated shortening pass (drop fillers, use concise synonyms, remove redundant pronouns)
   while preserving meaning, then re-check CPS.
9. **Smart Line Breaking**: Wrap cues to max 2 lines × 42 chars, balancing line lengths and
   breaking at the highest syntactic boundary.
10. **Sorani Post-Processing**: Apply Unicode normalization, RTL punctuation mirroring (`؟ ، ؛`),
    directional isolates (`\u2066...\u2069`) around embedded Latin names/numbers, and whitespace
    cleanup.
11. **QA & Confidence Scoring**: Compute per-cue confidence (0–100) combining model log-probs,
    length-ratio anomalies, CPS violations, glossary mismatches, and untranslated Latin tokens.
    Flag any cue with confidence < 70 or any QA warning.
12. Emit translated cues + per-cue metadata + project-level QA summary.

# PHASE 5 — BACKEND API & WORKERS (`api/`)
Build an async FastAPI backend with PostgreSQL 16 + Celery + Redis:
- **Database Tables**: `users`, `projects`, `media_files`, `cues`, `cue_revisions`,
  `glossary_terms`, `jobs`, `model_versions`, `review_events`. Store BOTH `machine_text` and
  `human_text` on every cue to power the retraining corpus.
- **Auth Endpoints**: `POST /api/auth/register`, `POST /api/auth/login`,
  `POST /api/auth/refresh`, `POST /api/auth/logout`, `GET /api/auth/me`.
- **Dashboard & Projects**:
  - `GET /api/dashboard/summary` (KPIs, recent projects, active jobs)
  - `GET /api/projects` (search, status filter, pagination), `POST /api/projects`
  - `GET /api/projects/{id}`, `PATCH /api/projects/{id}`, `DELETE /api/projects/{id}`
  - `POST /api/projects/{id}/upload` (multipart subtitle + optional video upload)
  - `POST /api/projects/{id}/translate` (enqueues Celery job → `job_id`)
- **Cues & Editor**:
  - `GET /api/projects/{id}/cues` (paginated + filters: `all`, `flagged`, `low_confidence`,
    `cps_violation`, `glossary_mismatch`, `unreviewed`, `edited`, plus text search)
  - `PATCH /api/cues/{cue_id}` (updates `human_text`, timestamps, or status; logs `cue_revisions`)
  - `POST /api/cues/{cue_id}/approve`, `POST /api/projects/{id}/cues/bulk-approve`
  - `POST /api/cues/{cue_id}/compress` (on-demand AI shortening for a single cue)
  - `POST /api/cues/{cue_id}/retranslate` (with optional custom context/glossary hint)
- **Glossary & QA**:
  - `GET/POST /api/projects/{id}/glossary`, `PATCH/DELETE /api/glossary/{term_id}`
  - `POST /api/projects/{id}/glossary/extract` (auto-NER), `POST /api/projects/{id}/glossary/import`
  - `GET /api/projects/{id}/qa`, `POST /api/projects/{id}/qa/auto-fix`
- **Export & Stats**:
  - `GET /api/projects/{id}/export?format=srt|vtt|ass|json|txt&encoding=utf-8|utf-8-sig&bilingual=false`
  - `GET /api/projects/{id}/stats` (edit rate, avg confidence, CPS violations)
- **Jobs & SSE**:
  - `GET /api/jobs`, `GET /api/jobs/{job_id}`, `POST /api/jobs/{job_id}/cancel`
  - `GET /api/jobs/{job_id}/stream` (SSE real-time progress %, stage name, ETA, and log lines)
- **Models & Feedback Loop**:
  - `GET /api/models` (versions + evaluation metrics), `POST /api/models/{id}/activate`
  - `GET /api/models/feedback-stats`, `POST /api/feedback/export` (approved edits → training JSONL)
  - `POST /api/models/train` (enqueues Celery LoRA fine-tuning job on new human edits)
  - `POST /api/models/compare` (runs two model versions on sample lines for A/B comparison)
- **Operational Requirements**: Request validation, rate limiting, structured logging, Prometheus
  `/metrics`, `/health` checks, graceful shutdown, Alembic migrations, and `api/db/seed.py`.

# PHASE 6 — COMPLETE MULTI-PAGE FRONTEND (`web/`)
Build a responsive, dark/light-mode multi-page Next.js App Router application.

### Layout & Navigation Architecture
1. **Public Pages (`(marketing)` & `(auth)`)**:
   - `/` (**Landing Page**): Hero explaining English ↔ Sorani film subtitling, live interactive
     3-line demo translator widget, feature breakdown (RTL precision, CPS control, glossary),
     and Login/Register CTAs.
   - `/login`, `/register`, `/forgot-password`: Form validation, error banners, loading spinners,
     and automatic redirect to `/dashboard` when authenticated.
2. **Authenticated App Shell (`(app)/layout.tsx`)**:
   - Protected by `middleware.ts` (redirects unauthenticated requests to `/login`).
   - **Collapsible Left Sidebar**: Links to `/dashboard`, `/projects`, `/projects/new`, `/jobs`,
     `/models`, `/settings`. When inside `/projects/[id]/*`, expands a contextual project sub-menu
     (Overview, Subtitle Editor, Video Studio, Glossary, QA Report, Export).
   - **Persistent Top Header**: Breadcrumb trail, active background jobs indicator with live SSE
     progress popover, Dark/Light theme toggle, UI language toggle (English / Kurdish Sorani UI),
     and user profile menu.

### Dedicated Application Pages (Every route must be a standalone `page.tsx` file)
3. **`/dashboard` (Workspace Home)**:
   - 4 KPI stat cards (Total Films, Cues Translated, Avg Confidence, Pending Review Cues).
   - Quick-start dropzone card ("Drop an .srt file to start translating") + Recent Projects grid +
     Active Jobs live widget.
4. **`/projects` (Projects Directory)**:
   - Grid/Table view switcher, live search, status filter (`All`, `Translating`, `Needs Review`,
     `Completed`), sort by date/confidence, pagination, and delete confirmation modal.
5. **`/projects/new` (4-Step Upload & Configuration Wizard)**:
   - Step 1: Drag-and-drop `.srt`/`.vtt`/`.ass` + optional `.mp4`/`.mkv`/`.webm` video; live
     preview of parsed cue count and duration.
   - Step 2: Direction (`EN → CKB` or `CKB → EN`), AI Model Version picker, Target CPS slider
     (default 17), Max chars/line (default 42), Register (`Formal` / `Informal` / `Auto`),
     Profanity softening toggle (OFF by default).
   - Step 3: Initial Glossary CSV upload or quick character-name entry.
   - Step 4: Summary card → "Start AI Translation" → redirects to `/jobs/[jobId]`.
6. **`/projects/[id]` (Project Overview & Analytics)**:
   - Film metadata card, pipeline completion progress, confidence score histogram, CPS violation
     breakdown, recent edit activity feed, and action cards linking to Editor, Video Studio,
     Glossary, QA, and Export.
7. **`/projects/[id]/editor` (Full-Screen Synchronized Subtitle Studio)**:
   - Collapsible app sidebar for maximum workspace width.
   - Top Filter & Action Bar: Filter pills (`All`, `Flagged`, `Low Confidence`, `CPS Violation`,
     `Glossary Mismatch`, `Unreviewed`), Find & Replace drawer, Bulk Approve button, Undo/Redo.
   - Center Virtualized Two-Column Table:
     - Left column: Cue index, start/end timestamps, duration, speaker badge, English text (`dir="ltr"`).
     - Right column: Editable Sorani textarea (`dir="rtl"`, `Vazirmatn` font) with `<bdi dir="ltr">`
       isolation, live CPS badge (green ≤ 17, amber 17–21, red > 21), line/char counter,
       confidence color bar, and Approve button.
     - Keyboard navigation: `↑`/`↓` move rows, `Enter` focus editor, `Ctrl+Enter` approve & jump
       to next flagged cue, `Ctrl+S` force save, autosave with optimistic updates.
   - Right Context & AI Inspector Sidebar: Surrounding context (previous 3 & next 3 lines),
     matched glossary terms, QA warnings, one-click **"AI Shorten (Fix CPS)"** button,
     **"Re-translate"** button, and `machine_text` vs `human_text` revision diff.
8. **`/projects/[id]/preview` (Video & Timing Sync Studio)**:
   - HTML5 video player with custom RTL Kurdish Sorani subtitle overlay renderer (accurate
     Arabic-script shaping, customizable preview font size & background box).
   - Interactive cue timeline below the player: click any cue to seek video, `±100ms` timestamp
     nudge controls, and inline text editing while watching.
9. **`/projects/[id]/glossary` (Project Glossary Manager)**:
   - Searchable table (`English Source`, `Sorani Target`, `Category`, `Occurrences`, `Locked` switch).
   - Actions: "Auto-Extract Names with AI", "Add Term" modal, CSV/JSON bulk import & export, and
     "Re-check Script Against Glossary".
10. **`/projects/[id]/qa` (Quality Assurance Audit Page)**:
    - Summary cards by issue type (CPS Violations, Line Length > 42, > 2 Lines, Untranslated Latin
      Words, Glossary Mismatches, Timing Overlaps).
    - Filterable list of flagged cues with **"Auto-Fix All Compressible Cues"** button and
      direct **"Open in Editor"** deep-links (`/projects/[id]/editor?cueId=...`).
11. **`/projects/[id]/export` (Export & Packaging Center)**:
    - Format cards (`.srt`, `.vtt`, `.ass` with font/color/margin styling options, `.json`, `.txt`).
    - Encoding selector (`UTF-8` vs `UTF-8 with BOM`), Bilingual dual-subtitle toggle, Approved-only
      vs All cues toggle, live output preview pane, and Download button.
12. **`/jobs` & `/jobs/[jobId]` (Jobs Queue & Live Execution Monitor)**:
    - `/jobs`: Table of all translation & training jobs with status badges, progress bars,
      duration, and cancel/retry buttons.
    - `/jobs/[jobId]`: Live SSE-powered stage stepper (Parsing → Reconstruction → NER Glossary →
      Translation → Compression → QA), real-time streaming log console, and link to finished project.
13. **`/models` (AI Models & Continuous Training Hub)**:
    - Registered model versions table (`nllb-600m-zeroshot`, `nllb-600m-sorani-lora-v1`, etc.)
      with `chrF++`, `spBLEU`, `COMET`, and CPS violation metrics on `gold_test.jsonl`, plus
      "Set as Active Model" button.
    - **Human Feedback Retraining Panel**: Count of approved human edits ready for training,
      "Export Training JSONL" button, and **"Launch LoRA Fine-Tuning Job"** modal (configure
      epochs, learning rate, base adapter → launches Celery training job).
    - **Live A/B Model Comparison**: Input 1–5 English subtitle lines (with context) and compare
      outputs from two model versions side by side.
14. **`/settings` (User & Workspace Settings)**:
    - Profile & password update, default subtitle preferences (default CPS cap, max chars/line,
      default export encoding), and personal API token management.

### Strict RTL & CSS Rules
- Never mix `dir="ltr"` and `dir="rtl"` on the same DOM element.
- Use CSS logical properties (`ms-*`, `me-*`, `ps-*`, `pe-*`, `text-start`, `text-end`) across
  all components so the UI works cleanly in both English and Kurdish interface modes.
- Isolate any Latin proper nouns or numbers inside Sorani text with `<bdi dir="ltr">...</bdi>`.

# PHASE 7 — CONTINUOUS FEEDBACK LOOP & PACKAGING
1. Nightly/on-demand Celery task exporting approved human edits to training JSONL; `ml/train`
   supports continued LoRA fine-tuning from the active adapter; `model_versions` tracks cue lineage.
2. `docker-compose.yml` running `web`, `api`, `worker-cpu`, `postgres`, `redis`, and `minio`, plus
   `docker-compose.gpu.yml` for NVIDIA GPU inference/training.
3. `api/db/seed.py` creating a demo user, sample model version records, and a pre-loaded
   English ↔ Sorani film project for immediate exploration.
4. Automated test suite: `pytest` (Sorani normalizer, IoU aligner, CPS compressor, 12-step
   pipeline, FastAPI endpoints) + `vitest` + Playwright E2E test (login → upload → edit → export).
5. Comprehensive `README.md` covering setup, local dataset directory format, training & evaluation
   commands, Docker deployment, and troubleshooting Sorani fonts/RTL/UTF-8 BOM in media players.

# LEGAL & ETHICAL REQUIREMENTS
- Do not scrape or redistribute copyrighted subtitles. Data adapters must read only from a
  user-supplied local directory, and `README.md` must document dataset licensing responsibility.
- Mark machine-translated output in exported subtitle file metadata comments.
- Keep profanity intact by default; provide a configurable softening toggle that is OFF by default.

# EXECUTION & GITHUB PUSH PROTOCOL
Work phase by phase (Phase 0 through Phase 7). Do NOT skip files or summarize code.
At the end of EVERY phase:
1. List every file created or modified.
2. Run the verification/test commands for that phase and report the output.
3. Stage, commit with a descriptive Conventional Commit message, and push to the `AI_training`
   GitHub repository (`git add . && git commit -m "..." && git push -u origin main`).

Begin now with **Phase 0 (Initialize `AI_training` repo & GitHub remote)** and **Phase 1 (Data Preparation & Alignment)**.
