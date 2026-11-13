# Offline pair-overlap audit

This is a **review aid**, not an approval or a training gate. It compares supplied, local English–Sorani pair text across an existing subtitle `DatasetDict`'s train/validation/test splits and, optionally, against completed text-pair or TMX staging outputs. It can also check supplied direction-specific corrected test-gold references. It never downloads a corpus or model, starts or modifies a job, changes the inputs, creates splits, removes pairs, or makes staged data trainer-ready.

Only independently authored **synthetic fixtures** have been run through this audit. Before processing real material, establish the rights, privacy conditions and approval for *all* inputs and the intended processing. The script cannot establish them from a license label or user declaration. Do not mark real material as synthetic. No model or dataset has been cleared by this feature.

## Synthetic demo

From the repository root, in a Python 3.11 environment with the project dependencies:

```powershell
python -m ml.data.audit_overlap --demo --output-dir outputs/overlap-demo-01
```

The command creates a temporary synthetic staging set and a mock `DatasetDict`. The mock IDs and start values are test fixtures, **not actual works or subtitle timings**. It intentionally includes exact and near-text matches, so it writes a report and exits **1** to request review; this is expected. It needs no network, real corpus, model, GPU, Docker, database, or worker. Use a new output folder name each time. After reinstalling the project, `sorani-audit-overlap` is also available.

The output folder contains:

- `overlap-report.json`: input fingerprints, comparison counts, method, limits and caveats. `training_ready` is always `false`. A complete report with schema `sorani-pair-overlap-audit-v1` is written last; verify the detail-file hash before relying on it as a completion marker. It is not a legal, language, timing or quality approval.
- `matches.jsonl`: review leads with input labels, pair/film/source-group identifiers, signal names, and any applicable similarity scores. It does not deliberately copy pair text. It can still reveal sensitive information through identifiers or hashes—especially if someone put dialogue or names into an ID. Keep **both** files private and out of Git and shared storage. The output's default `outputs/` parent is Git-ignored, but custom paths are the operator's responsibility.

The details file is capped at **50,000 records**. If `details.truncated` is `true`, aggregate counts are still computed within the defined work limits, but the file is incomplete. Do not treat it as a complete list of items for review. The printed summary reports the truncation as well.

## Inputs and future authorized use

The required `--dataset-dir` is a trusted local Hugging Face `DatasetDict` with exactly `train`, `val` and `test`, such as the **`dataset/` subdirectory** of the subtitle builder's output. `val` and `test` must be nonempty. If at least one staging directory is supplied, `train` may be empty, so general-text candidates can be checked against real held-out rows **without inventing training films or timings**. An empty-train dataset is not valid for the current trainer or evaluator; the audit does not create it or make it trainer-ready. Without staged candidates, `train` must be nonempty. All supplied timed records must satisfy the existing trainer's pair ID, film ID, start-time, provenance and language-flag shape checks. The audit uses film ID equality to report shared labels; different IDs are **not evidence of different works**. It does not verify that timings are real or positive durations, and it does not compute CPS. The evaluator's separate genuine-duration and reviewed-gold rules still apply. This checker is not a generic SMOL/CSV/SRT importer or a new text-only training contract.

`--staging-dir` is optional and may be repeated. It accepts the complete, unmodified review output directories of the [paired-line](text-pair-staging.md) and [TMX](tmx-pair-staging.md) tools. It checks their completion schema, manifest and output hashes, counts, source-group and row identity, and rejects duplicate staged pair IDs across inputs. At least one valid staged pair must be present in each supplied directory. Quarantined rows are **not** compared: their number is recorded as `quarantined_not_compared`. They must not silently be used as training data. The supplied staging-directory order identifies `staging-1`, `staging-2`, and so on in the reports. Staged pairs remain unverified and cannot be consumed by the current trainer.

Optionally supply `--gold-en-ckb` and/or `--gold-ckb-en` for the existing evaluator's **test** gold JSONL, using a separate file for each direction if their corrections differ. Each ID must identify a unique test pair, its film ID must match, and the direction's source text must be unchanged from the dataset, as in the evaluator. The corrected target is compared as an additional test variant; the whole original test split is still checked. Gold review status is counted, not verified. If a gold file is omitted, `gold_not_supplied` records that its corrected references were **not** checked. This tool does not create or approve a gold set.

Once the relevant rights and informed approval are established, an example using **your own authorized local paths** is:

```powershell
python -m ml.data.audit_overlap `
  --dataset-dir data/processed/subtitles/dataset `
  --staging-dir data/processed/text-pairs/collection-01 `
  --gold-en-ckb data/processed/review/gold-en-ckb.jsonl `
  --gold-ckb-en data/processed/review/gold-ckb-en.jsonl `
  --output-dir outputs/overlap-review-01
```

These are placeholders, not supplied files or permission to use a source. Omit a staging directory to compare only the subtitle dataset, and omit a gold option only with the corresponding limitation understood. Use trusted local dataset and JSON inputs. Choose a private parent directory: on Unix the new output is requested owner-only; on Windows it inherits the parent's ACLs. Input and output directories must not contain each other, and an existing output is never intentionally overwritten. The report pins the stage/gold bytes and the trainer-relevant dataset fingerprint; inputs are reread before publication to detect ordinary changes. Hashes are **not signatures**, do not establish authenticity or rights, and cannot defend against every concurrent or malicious modification. Do not modify inputs during the audit.

## What is compared

The comparisons are:

- dataset `train` vs `val`, dataset `train` vs `test`, and `val` vs `test`;
- each supplied staging set vs `val` and `test`;
- if a gold variant is supplied, dataset `train`, `val`, and each staging set vs that direction's gold records.

Staging sets are not checked against one another or against the training split; they are candidates for the same training side, not independently assigned splits. Quarantined, filtered-out and original raw subtitle material are outside this check. The checker does not compare staging source-group IDs to film IDs: these labels have different meanings and cannot establish independent works.

Each `pair_signal` detail links one left/right row pair. Signal counts overlap rather than adding up to a unique count:

- `same_pair_id` is an exact identifier match. The subtitle dataset's own pair IDs must already be unique across splits; reuse between a staged set and a held-out record is reported even if text differs.
- `english_exact` and `sorani_exact` compare the respective language fields. `bilingual_exact` means both matched in the same linked row pair. This detects the same stored pair regardless of whether a later model is run English→Sorani or Sorani→English. `swapped_bilingual_exact` additionally flags an exact pair with the two language fields swapped; that is a warning to inspect labeling, not proof of original translation direction.
- `english_near` / `sorani_near` are distinct, non-exact strings with **character-trigram set Jaccard ≥ 0.90** on that side. Both strings must have at least **12 normalized Unicode code points**. `bilingual_near` means each side is exact or near and at least one side is near. Scores for matched sides are recorded; exact is `1.0`. A short string is still checked for exact equality but **not** for near similarity.
- `shared_film_id` details report identical dataset film IDs across splits even when pair text differs. Shared labels require investigation; different labels do not prove independence or rights.

For matching, text is Unicode NFKC-normalized, case-folded and whitespace-collapsed. ZWNJ is retained, Arabic `ه` is not guessed, and punctuation is not stripped. These transformations can still change typography and must not be mistaken for bilingual normalization or verification. The near matcher uses a deterministic, rare-first prefix index and then verifies the exact Jaccard score, rather than sampling. It finds all matches **under this one metric and threshold** when the run completes, but that metric is not comprehensive: paraphrases, spelling, punctuation and invisible-formatting variations, different cue segmentation, reordered words, source/translation alternatives, and similar text below the threshold can be missed. Conversely common short lines, repeated character shingles, or unrelated text can produce signals without actual leakage. A qualified reviewer needs to interpret them.

The audit refuses nonempty explicit `source_context` rather than pretending to screen it. It also does **not** reconstruct or compare multi-cue context assembled later by the trainer; individual pair comparisons are not a complete context audit. It does not check unseen model-pretraining data, source documents, independent work ownership, derived feedback, unsupplied gold files or translations, or material excluded during dataset preparation. A `no_signals_detected` result only means no signals were found **within the supplied inputs and the defined method**. It is not proof that a model, dataset, work, split, or benchmark is uncontaminated or commercially usable.

The final test set must remain restricted and untouched for model selection. Have an appropriately authorized reviewer perform the pre-training contamination check, manage access to its potentially revealing reports, and resolve findings under a predefined protocol. Do not repeatedly tune the model or choose examples against the final test set. Reserve genuinely independent, rights-cleared works, review their provenance and translations, and use separate validation data for development.

## Exit codes and resource limits

- **0:** no signals detected within the limited scope above; not an approval.
- **1:** report written with one or more review signals, including shared film IDs. The synthetic demo intentionally returns 1.
- **2:** invalid, incomplete, changed, or unsupported input; a safety/resource limit; or an output error. No complete report is intentionally published. A crash can leave an incomplete output directory; inspect and remove it manually before retrying.

To avoid silently skipping comparisons, the implementation aborts if its fixed limits are exceeded: 200,000 dataset rows, 256 MiB of Arrow dataset content, 1,000,000 combined staged rows, 2 GiB of pair/quarantine outputs per staging input, a 4 MiB JSONL line, or 100,000 characters per text. The matching budget is 2,000,000 reference trigram windows, 100,000,000 query windows, 20,000,000 index-posting visits, and 5,000,000 candidate checks per run. These are work bounds, not guarantees of peak RAM. An error is **not** a clean result. If necessary, audit smaller candidate batches while preserving full held-out coverage and independently account for every batch; do not split or selectively omit the held-out set to obtain a reassuring number.

See [the commercial-use plan](sorani-data-import-plan.md) for the remaining rights, model, review and evaluation gates.
