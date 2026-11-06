# English ↔ Sorani subtitle training

This repository is being built in phases toward the product described in [SPEC.md](SPEC.md). **Phases 0 and 1 implement local data preparation only.** Model training, translation/inference, the API, the web app, Docker deployment, and continuous feedback are not implemented yet.

The Arena session uses the existing public `bryar-2005/Ai_trainig` repository and its fixed session branch; it does not reinitialize Git, create a separate `AI_training` remote, or push to `main`.

## Legal and privacy

Use only subtitle files, translations, and models that you are authorized to process. Do not scrape or redistribute copyrighted subtitles. The adapter reads only paths supplied on the local filesystem and does not download data. Dataset, gold-review, and model files may contain private or copyrighted content; keep them out of Git and storage you do not control. Adding a `license` field is documentation, not proof of permission.

## Setup

Requires Python 3.11 and a platform supported by `fasttext-wheel`. No GPU is needed for Phase 1.

```bash
python3.11 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
pytest -q
ruff check ml tests
```

NumPy is intentionally restricted to 1.x because the current `fasttext-wheel` prediction API is incompatible with NumPy 2.x. PyArrow is also pinned to the compatible 21.x series. `.env.example` contains optional cache settings, not secrets. The CLI does not require an `.env` file.

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

## Tests and limitations

`pytest -q` exercises all supported formats, normalization, 1:1/1:n/n:1 alignment, fastText language ID using a locally trained synthetic test classifier, filtering, MinHash, film-level splits, safe output handling, and a real CLI → `DatasetDict` → review-sample round trip. Fixtures are original test sentences, not movie subtitles. GitHub Actions runs Python lint and tests; frontend and model-regression CI will be added when those phases exist.

Phase 1 does not train or provide a translation model. Automated language ID, quality proxies, temporal matches, and positional spelling heuristics require human review before training or publishing results.
