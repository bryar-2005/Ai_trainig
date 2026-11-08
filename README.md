# English ↔ Sorani subtitle training

This repository implements the workflow described in [SPEC.md](SPEC.md). It includes local data preparation, context-aware LoRA training and evaluation, a subtitle inference/QA engine, a FastAPI/Celery backend, and a multipage Next.js review workspace. There is no supplied subtitle corpus, verified gold set, pretrained checkpoint, or trained translation model in this repository. Real translation requires authorized model assets and a running worker; the interface does not substitute canned output or claim model quality.

The Arena session uses the existing public `bryar-2005/Ai_trainig` repository and its fixed session branch; it does not reinitialize Git, create a separate `AI_training` remote, or push to `main`.

## Legal and privacy

Use only subtitle files, translations, and models that you are authorized to process. Do not scrape or redistribute copyrighted subtitles. The dataset adapter reads only paths supplied on the local filesystem and does not download data. Dataset, gold-review, tokenizer, training-log, and model files may contain private or copyrighted content; keep them out of Git and storage you do not control. Adding a `license` field is documentation, not proof of permission.

The default [NLLB-200 distilled 600M checkpoint](https://huggingface.co/facebook/nllb-200-distilled-600M) is provided under **CC-BY-NC 4.0**, which restricts commercial use; its model card describes it as a research model. Review the model card, license, your dataset permissions, and applicable obligations before training, distributing artifacts, or deploying. These restrictions are not removed by LoRA fine-tuning.

## Setup

Requires Python 3.11 and a platform supported by `fasttext-wheel`. Data preparation and the small offline training smoke test do not need a GPU.

```bash
python3.11 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
pytest -q
ruff check ml tests
```

NumPy is intentionally restricted to 1.x because the current `fasttext-wheel` prediction API is incompatible with NumPy 2.x. PyArrow is also pinned to the compatible 21.x series. `.env.example` contains optional cache settings, not secrets. The CLI does not require an `.env` file.

## Web workspace

See [docs/frontend.md](docs/frontend.md) for the Next.js setup, browser tests, account and translation workflow, accessibility, and security/deployment notes. With the API, PostgreSQL, Redis, and Celery running as described in [docs/backend.md](docs/backend.md), use Node.js 22 or newer:

```bash
cd web
npm ci
npm run dev
```

Open `http://localhost:3000`, register an account, and create a project from an authorized subtitle file. An administrator must register compatible model assets before running translation. The workspace supports draft editing and export without a model. Set `INTERNAL_API_ORIGIN` on the **Next server**, not in browser code, if the API is not at its default `http://127.0.0.1:8000`.

## Local parallel subtitle layout

Place authorized subtitle files in one folder per film. The folder name is its stable film ID; use exactly one `en` file and one `ckb` file per film. The two formats may differ, and supported extensions are `.srt`, `.vtt`, `.ass`, and `.ssa` (case-insensitive). Nested film directories and symlinked film folders/files are not followed.

```text
data/raw/
  film-one/
    en.srt
    ckb.srt
    metadata.json             # optional
  film-two/
    en.vtt
    ckb.ass
  film-three/
    en.ssa
    ckb.srt
```

Optional `metadata.json` applies to every pair from the film:

```json
{
  "provenance": "human",
  "quality_score": 0.9,
  "license": "Your actual license identifier or permission",
  "attribution": "Your actual source attribution"
}
```

Provenance must be `human`, `machine`, or `unknown` (default); quality must be between 0 and 1. Do not mark material human-translated without evidence. If quality is omitted, the unshifted temporal IoU is recorded as a **temporal proxy**, not as a semantic quality assessment. Original film ID, 1-based parser-order cue IDs, and per-cue timestamps, cleaned text, speakers, SFX and styles remain traceable, including for merged groups. Source file hashes, provenance, and license metadata are also recorded. Pure SFX/ad-only cues are counted but do not become text-training pairs.

## Build a dataset

For real language verification, supply a **local** fastText model whose labels include English (`en`/`eng`) and explicit Central Kurdish (`ckb`/`ckb_Arab`). The [fastText language identification documentation](https://fasttext.cc/docs/en/language-identification.html) describes the 176-language model; obtain it yourself after reviewing its license. The CLI neither downloads models nor scrapes subtitles. A generic `ku` label is not accepted as proof of Sorani.

```bash
python -m ml.data.build_dataset \
  --input-dir data/raw \
  --output-dir data/processed/subtitles \
  --lid-model data/cache/lid.176.ftz
```

A convenient entry point is also available as `sorani-build-dataset`. Use `--help` for all options. Language identification on short subtitle lines can be unreliable; inspect rejection counts and review accepted samples with a Sorani speaker. For a **development-only**, explicitly unverified run without a model:

```bash
python -m ml.data.build_dataset \
  --input-dir data/raw \
  --output-dir data/processed/subtitles \
  --skip-language-id
```

Script-only results have `language_verified=false`, null language labels, and an explicit report warning. Arabic-script ratio alone cannot distinguish Sorani from Persian or Arabic. `--lid-threshold` defaults to 0.3 and can be adjusted after evaluating your model on labeled examples; it is not a guarantee of correctness.

The command writes:

```text
data/processed/subtitles/
  dataset/                 # Hugging Face DatasetDict: train, val, test
  stats.json               # machine-readable statistics and processing diagnostics
  stats.md                 # human-readable summary
  .sorani-dataset-output   # safety marker for deliberate --overwrite
ml/eval/gold_test.jsonl    # held-out test sample for human verification
```

The output and gold paths must be outside the input directory; the gold path must also be outside the output directory. Use `--gold-path` to choose a different review file. Existing output is not overwritten unless `--overwrite` is supplied, and even then the output directory must carry the tool's safety marker. Gold and generated dataset files are ignored by Git under the default paths; subtitle and JSONL files are also ignored globally as a safeguard. Other custom output paths must also be kept private and out of Git.

Read the dataset locally:

```python
from datasets import load_from_disk

dataset = load_from_disk("data/processed/subtitles/dataset")
print(dataset["train"][0])
```

## Processing and assumptions

- HTML and ASS tags, known advertisement lines, and musical symbols are removed. Uppercase speaker prefixes, ASS names, VTT voice spans, and square-bracket sound effects are retained as metadata. Profanity is not softened.
- Pairs are chosen by a global, order-preserving temporal match. A match can be 1:1, 1:n, or n:1 (up to three cues by default). Each group's **original** start and end boundaries must each be within 500 ms, and its unpadded original-span IoU must be ≥ 0.5. The tool never shifts timestamps. Stricter boundaries can leave some valid translations unmatched; inspect the counts and adjust alignment strategy if needed.
- Nonempty, nonidentical pairs must have a Sorani/English length ratio of 0.4–2.5, and ≥80% of Sorani-side letters must be Arabic-script letters; spaces, digits, and punctuation do not enter the script denominator. A local fastText model then checks both language labels and confidence unless explicitly skipped.
- Normalization replaces `ك`→`ک`, `ي`/`ى`→`ی`, `ة`→`ە`, normalizes typography and Kurdish punctuation, preserves Kurdish-specific letters and ZWNJ, and removes input bidi formatting controls. **Arabic `ه` is ambiguous**: the default preserves and flags it. `--heh-policy positional` is an explicitly lossy positional heuristic (`ه`→`ە` at word end, otherwise `ھ`), not a reliable linguistic correction; review it manually.
- Bilingual character-trigram MinHash LSH proposes near-duplicates; exact per-language Jaccard must be ≥0.90 before deletion. Exact duplicates are always removed. LSH near-duplicate retrieval is probabilistic. Human-labeled pairs are preferred over unknown and machine pairs; this depends on truthful metadata.
- After deduplication, **whole films** are deterministically assigned to train/val/test, 80/10/10 by film count by default. Film sizes vary, so row ratios can differ. At least three retained films are required; `--allow-empty-splits` is for development only and is not a valid complete benchmark.
- Up to 1,000 pairs are sampled from the held-out **test split**, stratified by film and provenance, for review. The file is named `gold_test.jsonl` as requested, but each record is `review_status: "unverified"`: it is **not a verified gold benchmark until a person checks it**. If fewer than 1,000 pairs exist, all available test pairs are written once and the report warns about the shortfall.
- Reports include pair/film/provenance counts, per-film counts, token-length distributions, rejections, and OOV rates. OOV is defined as the share of case-folded regex word tokens absent from the **training split vocabulary**; it is not the NLLB tokenizer's unknown-token rate. Input file hashes support reproducibility.

## Train an English ↔ Sorani LoRA adapter

Install the optional training dependencies into the same Python 3.11 environment:

```bash
python -m pip install -e '.[dev,train]'
```

Choose a PyTorch 2.6 build compatible with your GPU and driver; for CPU-only installations you may first install the official CPU build from the [PyTorch installation instructions](https://pytorch.org/get-started/locally/). `.[dev,train,wandb]` adds optional Weights & Biases support. The default logger is local TensorBoard; enabling W&B can send run metrics to an external service, so review its privacy settings first.

The trainer consumes the Phase 1 `DatasetDict` with nonempty train and val splits, distinct films across train/val/test, unique pair IDs, provenance, and verified language labels. Run language identification and review the data first. `--allow-unverified-data` deliberately bypasses the verification requirement for exploration only and records a warning; it does not make data verified or suitable for a benchmark.

Start with an **offline smoke test** using your already prepared dataset:

```bash
sorani-train --dataset-dir data/processed/subtitles/dataset \
  --output-dir outputs/smoke-en-ckb --smoke-test --limit 200
```

The limit must be 200–500. If fewer examples exist, the command uses those available and records a warning. Smoke mode uses one epoch and a **tiny randomly initialized M2M100 model and locally built tokenizer**; it does not download or train NLLB, and its chrF++ score says nothing about translation quality. The tests exercise 200 synthetic train examples, 20 validation examples, evaluation, checkpointing, resume, and merge offline; the run completed within seconds on the development CPU. Runtime varies by hardware and dataset I/O.

For a real run, edit `ml/train/config.yaml` as needed and make the licensed NLLB checkpoint available from the Hugging Face Hub or set `base_model` to a local compatible checkpoint directory. Hub access and model weights are not provided by this repository. Set `local_files_only: true` to avoid Hub downloads, and optionally set `base_revision` to a specific Hub commit. Run separate adapters and output directories for each direction:

```bash
sorani-train --direction en-ckb --output-dir outputs/nllb-en-ckb
sorani-train --direction ckb-en --output-dir outputs/nllb-ckb-en
```

The implementation currently supports **one process and at most one visible GPU**. If your machine exposes multiple GPUs, select one first, for example `export CUDA_VISIBLE_DEVICES=0`. `mixed_precision: auto` selects bf16 on a compatible CUDA GPU, otherwise fp16 on CUDA or fp32 on CPU. GPU training and memory requirements have not been benchmarked here: monitor memory, reduce batch size/sequence limits if needed, and use accumulation to preserve an effective batch size. Real 600M-model training on CPU can be very slow. Check that warmup does not cover the whole run on a small dataset; the manifest warns when it does.

### Training details and artifacts

- Context is reconstructed from previous completed **source** sentences within each film, with speaker labels when known. A speaker change ends an unfinished utterance. The punctuation-based sentence heuristic is imperfect and should be audited; it does not reconstruct and re-split the current translated sentence, which belongs to the future inference pipeline. The first cue in a film has no previous context. Literal `<ctx>` in input is escaped, and context is separated from the current cue by the trained `<ctx>` token.
- The default is up to three previous sentences (configurable to three–five), and 256 source/target tokens. Under truncation, the current cue and newest context are prioritized. The manifest counts source, target, and context truncations; validation chrF++ is computed against the encoded, potentially truncated references.
- Loss is averaged per target example, then weighted human 1.5, unknown 1.0, machine 0.5; context is input only and is never included in the target. The example-weighted sum is normalized across the entire gradient-accumulation window, including a short final microbatch.
- Defaults are LoRA rank 16, alpha 32, dropout 0.05 on attention and feed-forward projections; 3 epochs, cosine learning rate 2e-4, 500 warmup steps, gradient checkpointing, and 8 accumulation steps. Validation uses generated translations and SacreBLEU chrF++ (`word_order=2`); the best validation checkpoint is reloaded, and early stopping is applied at evaluation steps.
- `training_manifest.json` records configuration, data fingerprint, direction, base revision or local-asset hash, token IDs, sample/provenance/truncation counts, metrics, and warnings, not source dialogue. A Hub revision is resolved to a commit and reused on resume/merge. For local bases, recognized weight, configuration, and tokenizer assets are hashed; changing them prevents resume/merge. Hashing a large local checkpoint takes additional I/O time.
- `checkpoint-*` directories contain the adapter, tokenizer, optimizer, scheduler, RNG and trainer/callback state; `adapter/` contains the best compact PEFT adapter and its trained `<ctx>` embedding row. These may contain private data or derived weights and must not be committed or shared without permission. Old checkpoints are pruned according to `save_total_limit`.

Resume the same run from its newest complete checkpoint, using **exactly the same configuration, paths, dataset, direction, and verification settings**:

```bash
sorani-train --direction en-ckb --output-dir outputs/nllb-en-ckb \
  --resume-from-checkpoint latest
# For a smoke run, also supply --smoke-test and the same --limit.
```

To create a standalone merged model (which requires the original base checkpoint and considerable disk space):

```bash
sorani-merge-adapter --adapter-dir outputs/nllb-en-ckb/adapter \
  --output-dir outputs/nllb-en-ckb-merged --device cpu
```

The merge command checks token IDs and local-base hashes, preserves the trained sentinel row, and records provenance in `merge_manifest.json`. An override of a Hub base with another path/repository cannot always be verified automatically; check `base_identity_verified` and ensure the weights are identical. Training outputs must be new/empty (unless resuming), must not contain the input dataset, and merge output must be new/empty and separate from the adapter. Default output and log locations are ignored by Git; custom paths are your responsibility.

## Evaluate a baseline against a trained adapter

Install the evaluation and training extras in the Python 3.11 environment:

```bash
python -m pip install -e '.[dev,train,eval]'
```

**Review the gold first.** Phase 1's `ml/eval/gold_test.jsonl` is a sample to review, not an already verified benchmark. A qualified bilingual reviewer should inspect each pair and its film context, correct the reference translation where needed, and set that record's `review_status` to `"verified"`. Keep the `pair_id`, `film_id`, and source-language text exactly as in the held-out dataset. For EN→CKB the source is `english` and the reference is `sorani`; for CKB→EN they are reversed. If corrections are needed on both sides, maintain direction-specific review files. The evaluator rejects missing/unverified reviews by default, unknown or duplicate IDs, changed sources, film leakage, and missing source durations. `--allow-unverified-gold` is an explicit **exploration-only** override; it is not human verification. Verify the language of any unverified context rows, too.

For a complete evaluation you must supply two assets that are **not included or obtained for you by this repository**:

1. A local SentencePiece `.model` for spBLEU. Obtain and verify the official FLORES-200 tokenizer yourself if you need FLORES-200-comparable results. An arbitrary SentencePiece model still gives SentencePiece-tokenized BLEU, but must not be presented as official FLORES-200 spBLEU. The report records the tokenizer's SHA-256 and SacreBLEU signature.
2. A trusted **reference-based** [COMET](https://github.com/Unbabel/COMET) checkpoint, including its companion `hparams.yaml` two directory levels above the `.ckpt`, and the checkpoint's pretrained encoder/tokenizer assets. A typical layout is `data/cache/comet/hparams.yaml` and `data/cache/comet/checkpoints/model.ckpt`. Obtain assets with appropriate permission and inspect their license. COMET checkpoints are PyTorch artifacts that can execute code when loaded: **do not load untrusted checkpoints**. COMET or Transformers may fetch encoder/tokenizer assets when online access is allowed. For offline evaluation, ensure the base model and COMET encoder/tokenizer are already available locally and pass `--local-files-only`.

COMET support and calibration for Sorani must be established independently. For example, the common `Unbabel/wmt22-comet-da` model card lists Kurdish Kurmanji but does not establish Sorani coverage. Treat any COMET score as an auxiliary, potentially unreliable signal, not proof of translation quality.

Supply a JSON glossary if you want adherence measured. For example, `data/cache/eval-glossary.json` can contain:

```json
[
  {"english": "Bridge", "sorani": "پرد", "locked": true},
  {"english": "Station", "sorani": "وێستگە", "locked": false}
]
```

Run the reviewed test set against the same dataset and base model used for the adapter:

```bash
sorani-evaluate \
  --dataset-dir data/processed/subtitles/dataset \
  --gold ml/eval/gold_test.jsonl \
  --direction en-ckb \
  --adapter-dir outputs/nllb-en-ckb/adapter \
  --spm-model data/cache/flores200-spm.model \
  --comet-checkpoint data/cache/comet/checkpoints/model.ckpt \
  --glossary data/cache/eval-glossary.json \
  --local-files-only --device cpu \
  --output-dir outputs/eval-en-ckb
```

Choose `--device cuda` if an appropriate CUDA GPU is available. The adapter manifest's direction, dataset fingerprint, base identity, language IDs, and `<ctx>` ID are checked. When the original local base was moved, `--base-model /path/to/copy` accepts a copy only if its recognized assets match the recorded hash. The baseline is the unadapted base with the **same** context formatting and generation settings as the adapter. The base does not receive the adapter's trained `<ctx>` embedding: the row may be newly random or previously unused padding, and is not necessarily pretrained as a context delimiter. The baseline is therefore not neutral with respect to the delimiter; interpret comparisons accordingly. Both use up to the adapter's configured 3–5 previous completed source sentences. The HTML displays up to five *available* sentences, which may include more than the model's configured window.

Alternatively, provide two prediction files, with exactly one JSON object per gold pair, in any order:

```json
{"pair_id":"held-out-pair-id","text":"the system's translation"}
```

Use `--baseline-predictions path/baseline.jsonl --candidate-predictions path/candidate.jsonl` instead of `--adapter-dir`. The generated combined `predictions.jsonl` can also be passed as both inputs on a later run; the appropriate `baseline` or `candidate` field is used. Precomputed mode cannot verify the models' origin or their use of context. For incomplete development-only runs, explicitly pass `--skip-spbleu` and/or `--skip-comet`; skipped metrics are null and warnings appear in the report. Random Phase 2 smoke adapters additionally require `--allow-smoke-model`, and their results do not measure translation quality.

Each evaluation requires a new or empty output directory separate from all inputs. It writes:

- `metrics.json`: both systems' scores, the chrF++ difference, configuration/asset hashes and warnings. The training-relevant dataset hash and an additional held-out timing hash are recorded separately.
- `predictions.jsonl`: baseline and candidate hypotheses for **every** gold pair.
- `comparison.html`: a self-contained, escaped side-by-side report. It shows up to 100 deterministic sampled pairs by default, with highlighted baseline→candidate differences; `--report-lines` can change the limit and `--label` can identify the run. Open the file in a browser.

These outputs can contain private dialogue and predictions; keep them local and out of Git. COMET's checkpoint and `hparams.yaml` hashes are recorded, but separately resolved encoder/tokenizer files are not fully hashed; retain the exact assets and environment for reproducibility. The process exits **1 after writing the report** when candidate chrF++ is lower than baseline, 0 when it is not, and 2 on input/evaluation errors. Passing the chrF++ gate alone is not proof the model is suitable for deployment.

### Metric definitions

- **chrF++:** corpus SacreBLEU chrF with word order 2, on a 0–100 scale; this is the regression gate. **spBLEU:** SacreBLEU corpus BLEU on SentencePiece-tokenized text, 0–100. **COMET:** the reference-based checkpoint's corpus system score; its scale and coverage depend on that model.
- **CPS:** visible Unicode grapheme clusters, including spaces but excluding line breaks and invisible controls, divided by the original *source aligned-group* duration. Report percentages above 17 and 21 CPS. This is a proxy: a 1:n alignment can span multiple subtitle cues, so it is not final per-cue QA.
- **Glossary adherence:** for each scored pair and locked term found in its source, check whether the target term appears in the hypothesis. Matching uses NFC normalization, case folding, and Unicode word boundaries; it does not perform morphological analysis. The rate is null when no locked term applies or no glossary was supplied.
- **Length ratio:** hypothesis visible-grapheme count divided by reference count (minimum denominator 1), summarized with min, mean, median, 90th percentile, and max.

## Translate and check a local subtitle file

The Phase 4 engine consumes an authorized local `.srt`, `.vtt`, `.ass`, or `.ssa` file and produces an auditable **JSON result**, not a finished subtitle export or a web app. Install `.[train]` and make a compatible NLLB base and, preferably, a Phase 2 adapter available. Weights are not provided. Use only trusted model, tokenizer, and NER assets; some checkpoint formats can execute code when loaded. For example:

```bash
sorani-translate \
  --input data/raw/film-one/en.srt \
  --output outputs/film-one-ckb.json \
  --direction en-ckb --model-version-id nllb-en-ckb-v1 \
  --adapter-dir outputs/nllb-en-ckb/adapter \
  --glossary data/cache/eval-glossary.json \
  --local-files-only --device cpu
```

Use a matching `ckb-en` adapter and source file for the reverse direction. `--model /path/to/base` can point to an identical copy of an adapter's original local base; its assets must match the training hash. Instead of `--adapter-dir`, `--model` may name a local base or merged model directory, or a Hub model if online access is permitted. `--model-revision` accepts an immutable 40-character Hub commit for base-only inference; adapters use their recorded revision. The result records model identity where available. A base-only model that needs a new `<ctx>` embedding has an **untrained** marker and may perform poorly; a trained adapter or merged model is preferable. `--allow-smoke-model` exists only for random smoke-test adapters: they cannot translate meaningfully. Choose `--device cuda` only with an available compatible GPU.

A glossary is optional; it uses the English/Sorani JSON array format shown above. Only user-locked entries are enforced. The default entity extractor proposes approximate spellings for capitalized English names and speaker labels, but cannot reliably classify people, places, or organizations. Proposals are **unlocked** and need human correction. `--ner-model /path/to/token-classification-model` uses a real Transformers NER model with PER/LOC/ORG labels, when one is supplied; review its license and accuracy. In CKB→EN, source-side English NER can only see English/Latin material present in the source or speaker labels—it is not Sorani NER. No NER assets are downloaded by this repository when `--local-files-only` is set.

The engine reports progress through 12 numbered stages, then produces original cue IDs and millisecond timestamps, normalized source offsets into reconstructed sentences, speaker/SFX/style metadata, generated `machine_text`, empty `human_text`/revision history, a model version ID, sentence-level selected-token log-probabilities, and cue/project QA. The CLI prints progress JSON lines to stderr and a summary to stdout; the JSON output may contain private dialogue and must stay out of Git. The output file must be new and outside input model directories. There is deliberately no `--overwrite` option.

Important limitations and review rules:

- Sentence boundaries and named entities are heuristic. Adjacent incomplete cues are merged, except across empty cues or explicit speaker changes. Offsets refer to normalized, space-joined source text; the individual cue timings never move. The previous 3 sentences are supplied by default (`--context-sentences` allows 3–5). Model limits can truncate source/context; this is flagged. Locked-term hints use the same context format, and a failed hard-constraint retry is followed only by replacement of an actual literal source term—not arbitrary insertion.
- Target text is re-split only at whitespace, using source-span proportions and preferred syntactic boundaries. When there are too few target words or no legal line break, the engine preserves words, leaves some cues empty or lines long, and flags them instead of splitting a word or shifting time.
- Reading speed defaults to 17 CPS with a 21 CPS hard-cap warning. A small conservative set of filler, repeated-pronoun, and concise-phrase rules attempts compression while protecting present locked terms. It cannot guarantee unchanged meaning or force an impossible cap; edits and unresolved violations are flagged for a human. Line wrapping similarly never truncates. `--target-cps`, `--hard-cps`, `--max-chars`, and `--max-lines` are configurable.
- Sorani normalization maps Arabic variants to Kurdish codepoints and uses `؟ ، ؛`; embedded Latin and numbers are wrapped in Unicode LRI/PDI isolates. Ambiguous Arabic `ه` is preserved and flagged, not guessed. Profanity is not softened by default; the API's optional user-specified replacements are documented in the backend guide. SFX annotations are retained in the source language, appended to the cue when necessary, and flagged for review.
- `--register auto|formal|informal` **flags** potentially inconsistent explicit Sorani pronouns for a speaker; it does not rewrite grammar or establish the speakers' relationship. QA also flags missing translations, glossary mismatches, source truncation, untranslated script, original timing overlaps/out-of-order cues, and low confidence. Confidence is an uncalibrated heuristic, not the probability that a translation is correct. Token log-probabilities are measured from the model's selected raw sequence by a memory-bounded teacher-forced pass; later glossary substitutions and compression do not change those scores.

The model-independent `BaseTranslator` interface and progress callback can be used by later workers. For example:

```python
from ml.inference.nllb_translator import NllbTranslator
from ml.inference.pipeline import PipelineOptions, translate_subtitles

model = NllbTranslator(
    direction="en-ckb", model_version_id="nllb-en-ckb-v1",
    adapter_dir="outputs/nllb-en-ckb/adapter", local_files_only=True,
)
result = translate_subtitles(
    "data/raw/film-one/en.srt", model, options=PipelineOptions(),
    progress=lambda event: print(event["stage"], event["percent"]),
)
```

A callback may raise an exception to stop the operation; the programmatic function itself writes no output file.

## Backend API and workers

The FastAPI/PostgreSQL backend and Celery/Redis workers support private projects, review, exports, resumable job progress and continued training. See **[backend setup and operational guidance](docs/backend.md)** for installation, migrations, administrator bootstrap, API security, storage, model registration, feedback safeguards and tests. The real multipage Next.js application is described in **[frontend.md](docs/frontend.md)**; CPU/GPU Compose, TLS, model provisioning, privacy and backups are covered in **[deployment.md](docs/deployment.md)**. No pretrained weights, real subtitle corpus or quality results are included.

## Tests and limitations

`pytest -q` exercises all supported formats, normalization, 1:1/1:n/n:1 alignment, synthetic fastText language ID, filtering, MinHash, film-level splits, safe output handling, and a real CLI → `DatasetDict` → review-sample round trip. With `.[train]` installed, it also checks context/speaker handling, loss weighting and gradient accumulation, a 200-example offline smoke run in both directions, exact CPU checkpoint resume, sentinel-row reload, and merged-model logits. The evaluation tests cover review and split validation, both CLI exit paths, HTML escaping and 100-row sampling, real SentencePiece tokenization, and adapter generation. With `.[eval]`, they also construct a tiny random local COMET checkpoint and call its actual loading and prediction APIs offline. Inference tests cover all stages, timing and lineage invariants, glossary retry, compression, RTL/QA, actual local tiny adapter/merged generation and token scores, and a tiny local Transformers NER model. Optional-dependency tests are skipped if their packages are absent. All fixtures are synthetic, not movie subtitles. GitHub Actions runs data checks and a separate offline CPU model job.

No real NLLB fine-tuning, pretrained subtitle inference, GPU benchmark, translation-quality evaluation, or reviewed gold benchmark has been completed in this environment. Automated language ID, quality proxies, temporal matches, sentence reconstruction, compression, named entities, and spelling heuristics require human review before using or publishing results.
