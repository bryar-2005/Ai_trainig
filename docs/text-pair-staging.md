# Local English–Sorani text-pair staging

This is a **review-only, offline CLI**, not an automatic training-data import into the web app. It reads two local UTF-8 files containing matching lines, records their source metadata, and produces a private staging report. It neither downloads a corpus nor starts or changes a job, model, or training run. No real corpus has been processed by us with this feature.

The existing timed-subtitle dataset builder remains separate. The current trainer and web UI **cannot consume these staged text pairs**. The output has no invented film IDs, timings, speakers, or reading-speed measurements, and is not a Hugging Face `DatasetDict`. Do not alter it to bypass the existing trainer's validation. Additional text-only data, split, evaluation and commercial-model work is required before training.

## Try it with synthetic examples

From the repository root in a Python 3.11 environment, run:

```powershell
python -m ml.data.stage_text_pairs --demo --output-dir outputs/text-pair-demo-01
```

This creates three independently authored, synthetic examples in a temporary directory and stages them. It does not need a model, GPU, internet, Docker, database, or worker. The output folder must **not already exist**; use a different folder name on a second run. `outputs/` is ignored by this repository's Git configuration.

The command prints counts, not dialogue. The new folder contains:

- `pairs.jsonl`: staged rows with original and normalized English/Sorani, source line, deterministic pair and source-group identifiers, and review flags. These rows contain text; keep them private.
- `quarantine.jsonl`: line numbers, identifiers, and reasons for invalid pairs. It does not copy the dialogue; refer to the untouched original input files to inspect a rejected line.
- `source-manifest.json`: the exact supplied metadata file, including any additional conditions. Keep it private as well.
- `staging-report.json`: counts and reasons, source and rights declarations, input/output hashes, and limitations. This file is written last and serves as the completion marker only if it is valid, complete JSON with the expected schema. A report is **not** a legal, linguistic, alignment, or training-quality approval.

The demo tests software behavior only. Its examples are not verified training data and cannot demonstrate translation quality.

## Stage real files only after authorization

First establish the applicable rights for both the original text and translations, the desired commercial uses, any attribution or other obligations, and any restrictions on processing. A valid open license can permit commercial use without a separate written grant; if the terms or provenance are unclear, obtain the necessary permissions or exclude the material. Keep evidence and records private. Merely downloading a file, setting a metadata field, or seeing a dataset-level license does not establish these rights.

The importer requires a source manifest with an explicit user attestation and evidence references. **It cannot check whether these statements are true.** The example below is a template, not evidence or permission: replace every placeholder with accurate information only after the review. Never declare real content `synthetic` to bypass the gate. The sample `permitted_uses` records training only; include `commercial_validation` and/or `commercial_testing` only when those specific uses are also authorized. Keep separate, independently authorized works for evaluation.

```json
{
  "source_id": "replace-with-actual-collection-id",
  "source_revision": "replace-with-actual-release-or-immutable-revision",
  "source_reference": "replace-with-source-URL-or-contract-reference",
  "provenance": "unknown",
  "rights": {
    "status": "user_attested",
    "affirmed_by_user": true,
    "original_text_evidence": "replace-with-original-work-rights-evidence-reference",
    "translation_evidence": "replace-with-translation-rights-evidence-reference",
    "license_or_permission": "replace-with-applicable-license-or-agreement-reference",
    "attribution": "replace-with-required-attribution-or-documented-none",
    "permitted_uses": ["commercial_training"]
  }
}
```

`source_id` and `source_revision` are labels supplied by the operator; the importer does not verify them against the provider. The actual input SHA-256 hashes pin the bytes read, and the exact manifest is preserved. `provenance` must be `human`, `machine`, or `unknown`; choose `unknown` unless the origin is supported by evidence. For a synthetic fixture, the status is `synthetic` and the only permitted use is `synthetic_testing`. For an attested real source, supported use labels are `commercial_training`, `commercial_validation`, and `commercial_testing`. The importer cannot authenticate the reviewer or validate referenced rights documents.

When the inputs are authorized, put the **extracted, paired** English and Sorani text files and the completed manifest in a private location, for example beneath the Git-ignored `data/raw/` folder. Each physical line in one file must correspond to the **same numbered physical line** in the other. Then, from the repository root:

```powershell
python -m ml.data.stage_text_pairs `
  --english-file data/raw/authorized/en.txt `
  --sorani-file data/raw/authorized/ckb.txt `
  --source-manifest data/raw/authorized/source.json `
  --output-dir data/processed/text-pairs/collection-01
```

The `sorani-stage-text-pairs` entry point is also available after reinstalling the project. The command reads local files only: it does not fetch, unpack, or scrape OPUS. OPUS's processed Moses download is paired text rather than a ready-to-use timed-SRT set; this workflow does not establish its permission, dialect, alignment, or suitability. It expects extracted line-per-record text and does **not parse** archives, CSV, TMX, or SMOL-style JSONL; do not pass those formats as if they were paired line files. A separate [offline TMX staging CLI](tmx-pair-staging.md) supports plain-segment TMX without creating subtitle timings or a trainer-ready dataset.

## Checks and limitations

- The importer checks the two **physical line counts before filtering** and refuses mismatches or two empty files. It accepts LF or CRLF files, UTF-8 with an optional initial BOM, and checks file hashes again while processing. A changed input fails the whole run. A final newline does not create an extra line.
- Blank, invalid UTF-8, overly long, or control-character lines quarantine the **entire matching pair**. Valid pairs after a bad line remain aligned by their original line numbers. `--max-chars` defaults to 5,000 characters per side; adjust only after understanding the data.
- Raw input files are left unchanged, with SHA-256 hashes recorded. Staged rows also contain decoded original text, with line endings and an initial BOM removed, plus normalized text. Normalization preserves ZWNJ and ambiguous Arabic `ه`; it flags the latter rather than guessing. Text-only normalization can still change typography and punctuation; inspect both versions.
- It flags exact duplicate bilingual pairs, identical text, unusual length ratios, missing expected-script letters and ambiguous `ه`. These are heuristics only: duplicates are not removed, near-duplicates are not checked, and the flags do not prove language, semantic alignment, authorship or quality. Arabic script or the generic code `ku` alone is not proof of Sorani. A qualified bilingual reviewer must check the material.
- There is **no assumed original translation direction** and no inferred previous-sentence context. Every row with the same operator-supplied source ID shares one conservative source group, even across separate files or revisions; pair IDs additionally bind the revision, file hashes and line number. This importer does not create train/validation/test splits. Source IDs are not proof that collections or works are independent. Before training, use actual documented work boundaries and screen for exact and near-duplicate leakage across works, sources, and directions; unknown grouping must not be presented as film-disjoint.
- Outputs are never deliberately overwritten. On normal failure, temporary files are cleaned up. An interrupted process may leave an incomplete new output directory or partial report; require a valid, complete `staging-report.json` with schema `sorani-text-pair-staging-v1` and check the output hashes before treating it as complete. Inspect and remove an incomplete directory manually before retrying. On Unix the new directory is requested with owner-only permissions; on Windows it inherits access controls from its parent, so choose a private parent folder.
- The example output locations are ignored by Git. If other locations are used, the operator must keep source files, staged text, manifests, and reports out of Git and shared storage. Do not put dialogue in the manifest or in logs. Preserve source evidence and assess privacy and retention requirements.

A separate [offline pair-overlap audit](overlap-audit.md) can compare completed staged pairs with held-out rows in an existing subtitle `DatasetDict`. It uses limited exact and trigram-near signals and is not an independence or rights approval. Only synthetic inputs have been run through it.

For the remaining work and commercial-use gates, see [sorani-data-import-plan.md](sorani-data-import-plan.md).
