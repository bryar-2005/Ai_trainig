# Local English–Sorani TMX staging

This **offline, review-only** tool reads a local translation-memory exchange (TMX) file containing English and Central Kurdish text. It is useful for investigating plain-text localization data, such as Mozilla's Pontoon export, **after the relevant rights and provenance have been reviewed**. It does not download or fetch a corpus. It does not create subtitle files, train a model, change a job, or import anything into the web app. We have tested it with independently authored synthetic XML only; no real corpus has been staged with it.

The resulting pairs are **not trainer-ready**. They have no invented film IDs, timings, speakers, work boundaries, previous-dialogue context, or subtitle reading-speed metrics. The current trainer and web UI cannot consume them. Source-language tags and a rights attestation are evidence supplied by the source/operator, not proof of language, translation quality, original authorship, or legal clearance.

## Synthetic demo

In a Python 3.11 environment with the project dependencies installed, from the repository root:

```powershell
python -m ml.data.stage_tmx_pairs --demo --output-dir outputs/tmx-pair-demo-01
```

The demo creates three synthetic English–Sorani pairs in a temporary TMX file, stages them locally, and prints counts only. It needs no corpus, network, model, GPU, Docker, database, or worker. The output directory must **not already exist**. After reinstalling the project, `sorani-stage-tmx-pairs` is also available.

The private output contains:

- `pairs.jsonl`: decoded originals, normalized texts, source-unit identifiers and ordinal, reported language tags, and review flags. Keep this file private: it contains text.
- `quarantine.jsonl`: ordinal, pair ID and reasons for each rejected unit, without the text itself. One bad side quarantines the whole unit; later units retain their original ordinals.
- `source-manifest.json`: the exact user-supplied source/rights declaration, including additional conditions.
- `staging-report.json`: counts, source declarations, input and output SHA-256 hashes, and limitations. A valid, complete report with schema `sorani-tmx-pair-staging-v1` is written last; verify both output hashes. It is a completion marker, **not** a rights, quality, or training approval.

## Local material after rights review

First check the original text and translation rights, scope of permitted commercial training/validation/testing, attribution and other obligations, privacy, and processing conditions. A valid open license may permit commercial use without an email, but a repository's license label does not settle every upstream or model-related question. Exclude material whose applicable rights cannot be established; do not mark it synthetic to bypass the gate.

Use the same source/rights manifest described in [text-pair-staging.md](text-pair-staging.md#stage-real-files-only-after-authorization). Supply a stable source ID, immutable revision and real evidence references, and attest only to uses you have actually reviewed. The tool validates the declaration's structure but **cannot verify it**. Keep the raw TMX, manifest and evidence private. Once the material is authorized for the planned processing, an example local invocation is:

```powershell
python -m ml.data.stage_tmx_pairs `
  --tmx-file data/raw/authorized/localization.tmx `
  --source-manifest data/raw/authorized/source.json `
  --output-dir data/processed/text-pairs/localization-01
```

Paths are examples, not supplied files or an authorization to use a particular dataset. The command does not unpack archives or download anything. The separately documented paired-line CLI remains available for already-extracted aligned text; neither tool supports SMOL JSONL.

## Supported format and review behavior

- Accepts an unnamespaced **TMX 1.4** root with one header, a nonempty declared header `srclang`, and one body containing translation units. XML encoding is interpreted by the parser from the file's declaration/BOM. Input is limited to 256 MiB and nesting to 32 levels; `--max-chars` limits each decoded segment to 5,000 characters by default, up to 100,000.
- Each unit must contain exactly two variants, one explicitly tagged `en` and one `ckb` (case-insensitive, with optional language subtags such as `en-US` or `ckb-Arab-IQ`). A generic `ku`, unknown language, multiple candidates, missing or blank segment, overly long text, disallowed decoded controls or line separators, or unsupported structure quarantines the entire unit. Characters that make the XML itself invalid abort the run instead. These tags are not independent language verification.
- A segment containing actual nested TMX XML markup, such as `<ph>`, is quarantined rather than silently flattened. Escaped literal markup and placeholders inside segment text are retained and heuristically flagged for review, not removed. Standard unit/variant `prop` and `note` metadata is allowed but is not fully copied to the JSONL. DTDs, entity declarations and external entities are forbidden. Malformed XML, unsafe XML, unexpected text outside units, or an unsupported document structure aborts the run without publishing output.
- The source file is not modified or copied to the output. **Keep the exact original TMX**: its byte hash and 1-based unit ordinal let reviewers recover metadata that the staged rows do not copy. XML parsing resolves standard character escapes and normalizes XML line endings; the `original_*` fields preserve decoded segment text, not the original XML bytes. Multiline segments are retained and flagged. Normalization retains ZWNJ and ambiguous Arabic `ه`, and flags the latter instead of guessing.
- Each row stores the header and unit `srclang` values where supplied, variant tags and their original order. These are reported source-language declarations, not proof of which wording was originally authored first; `original_direction` therefore remains `unknown`. Do not manufacture ordering or context from adjacent TMX units.
- Duplicate normalized bilingual text and repeated source `tuid` values are **flagged, not deleted**. A `tuid` can be absent or repeated. Pair IDs bind the source group, source revision, exact TMX hash, and unit ordinal. Duplicate references use original ordinals; a repeated-`tuid` reference can point to an earlier quarantined unit. All units with the same operator-supplied source ID share one conservative group across files and revisions. The tool creates no train/validation/test splits and does not establish independent works or check near-duplicate leakage.
- The input is hashed before and during parsing; changed bytes abort publication. Outputs are created only in a new directory, without intentional overwrite. On ordinary errors temporary files are removed; after a crash inspect and remove any incomplete output directory manually before retrying. The directory is requested owner-only on Unix; Windows inherits its parent's ACLs. Choose a private parent and keep dialogue, source files, manifests, and reports out of Git and shared storage.

A separate [offline pair-overlap audit](overlap-audit.md) can compare completed staged pairs with held-out rows in an existing subtitle `DatasetDict`. It uses limited exact and trigram-near signals and is not an independence or rights approval. Only synthetic inputs have been run through it.

For the source investigation, rights gates, required independent subtitle evaluation, and model integration work, see [sorani-data-import-plan.md](sorani-data-import-plan.md).
