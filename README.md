# English ↔ Sorani subtitle training

This repository is being built in phases toward the product described in [SPEC.md](SPEC.md). **Phases 0–2 implement local data preparation and a context-aware LoRA training pipeline.** There is no supplied subtitle corpus, verified gold set, pretrained checkpoint, or trained translation model in this repository. Translation/inference, standalone evaluation, the API, the web app, Docker deployment, and continuous feedback are not implemented yet.

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

## Tests and limitations

`pytest -q` exercises all supported formats, normalization, 1:1/1:n/n:1 alignment, synthetic fastText language ID, filtering, MinHash, film-level splits, safe output handling, and a real CLI → `DatasetDict` → review-sample round trip. With `.[train]` installed, it also checks context/speaker handling, loss weighting and gradient accumulation, a 200-example offline smoke run in both directions, exact CPU checkpoint resume, sentinel-row reload, and merged-model logits. Without the training extra, those optional tests are skipped. Fixtures are original synthetic sentences, not movie subtitles. GitHub Actions runs data checks and a separate offline CPU training job.

No real NLLB fine-tuning, GPU benchmark, translation-quality evaluation, or reviewed gold benchmark has been completed in this environment. Automated language ID, quality proxies, temporal matches, sentence reconstruction, and spelling heuristics require human review before using or publishing results.
