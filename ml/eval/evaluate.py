"""Compare a zero-shot base and fine-tuned adapter on reviewed held-out pairs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from ml.eval.compare import render_comparison
from ml.eval.metrics import compute_metrics, load_comet, load_glossary
from ml.eval.records import file_sha256, load_evaluation_records, load_predictions


def _output_path(path: Path, inputs: list[Path]) -> Path:
    requested = path.expanduser()
    if requested.is_symlink():
        raise ValueError("Evaluation output may not be a symlink")
    target = requested.resolve()
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError("Evaluation output must be a new or empty directory")
    for source in inputs:
        resolved = source.expanduser().resolve()
        if (
            resolved == target
            or resolved.is_relative_to(target)
            or (resolved.is_dir() and target.is_relative_to(resolved))
        ):
            raise ValueError("Evaluation output must not overlap an input file or directory")
    return target


def _write_outputs(
    target: Path,
    summary: dict[str, Any],
    html: str,
    pair_ids: list[str],
    baseline: list[str],
    candidate: list[str],
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{target.name}.evaluate-", dir=target.parent) as temp:
        stage = Path(temp) / "report"
        stage.mkdir()
        (stage / "metrics.json").write_text(
            json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        (stage / "comparison.html").write_text(html, encoding="utf-8")
        with (stage / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for pair_id, first, second in zip(pair_ids, baseline, candidate, strict=True):
                handle.write(
                    json.dumps(
                        {"pair_id": pair_id, "baseline": first, "candidate": second},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        if target.is_symlink() or (
            target.exists() and (not target.is_dir() or any(target.iterdir()))
        ):
            raise ValueError("Evaluation output changed or is no longer empty")
        if target.exists():
            target.rmdir()
        os.replace(stage, target)


def evaluate(
    *,
    dataset_dir: Path,
    gold_path: Path,
    output_dir: Path,
    direction: str,
    adapter_dir: Path | None = None,
    baseline_predictions: Path | None = None,
    candidate_predictions: Path | None = None,
    sentencepiece_model: Path | None = None,
    comet_checkpoint: Path | None = None,
    skip_spbleu: bool = False,
    skip_comet: bool = False,
    glossary_path: Path | None = None,
    allow_unverified_gold: bool = False,
    allow_smoke_model: bool = False,
    base_model_override: str | None = None,
    local_files_only: bool = False,
    device: str = "cpu",
    batch_size: int = 4,
    report_lines: int = 100,
    label: str | None = None,
) -> dict[str, Any]:
    if direction not in {"en-ckb", "ckb-en"} or device not in {"cpu", "cuda"}:
        raise ValueError("Invalid direction or device")
    if label is not None and (not isinstance(label, str) or not label.strip() or len(label) > 200):
        raise ValueError("A report label must contain 1–200 characters")
    if batch_size < 1 or not 1 <= report_lines <= 1000:
        raise ValueError("batch_size must be positive and report_lines must be 1–1000")
    if (adapter_dir is None) == (baseline_predictions is None or candidate_predictions is None):
        raise ValueError("Provide an adapter OR both baseline and candidate prediction files")
    if adapter_dir is not None and (
        baseline_predictions is not None or candidate_predictions is not None
    ):
        raise ValueError("Adapter and precomputed prediction modes are mutually exclusive")
    if (sentencepiece_model is None) != skip_spbleu:
        raise ValueError("Supply --spm-model or explicitly use --skip-spbleu, but not both")
    if (comet_checkpoint is None) != skip_comet:
        raise ValueError("Supply --comet-checkpoint or explicitly use --skip-comet, but not both")
    if base_model_override and adapter_dir is None:
        raise ValueError("--base-model is only valid with --adapter-dir")
    inputs = [dataset_dir, gold_path]
    inputs.extend(
        item
        for item in (
            adapter_dir,
            baseline_predictions,
            candidate_predictions,
            sentencepiece_model,
            comet_checkpoint,
            glossary_path,
        )
        if item is not None
    )
    base_source = base_model_override
    if adapter_dir and not base_source:
        manifest_path = adapter_dir / "training_manifest.json"
        if manifest_path.is_file():
            try:
                manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Cannot read adapter manifest: {exc}") from exc
            if isinstance(manifest_value, dict):
                base_source = manifest_value.get("actual_base_model")
    if isinstance(base_source, str) and Path(base_source).expanduser().is_dir():
        inputs.append(Path(base_source))
    target = _output_path(output_dir, inputs)
    records, data_info = load_evaluation_records(
        dataset_dir, gold_path, direction=direction, allow_unverified_gold=allow_unverified_gold
    )
    warnings: list[str] = []
    if data_info["unverified_gold_count"]:
        warnings.append("Gold contains unverified references; these results are exploratory.")
    if data_info["language_unverified_test_pairs"]:
        warnings.append(
            "The test dataset contains language-unverified pairs; manually review both the "
            "scored pairs and their source context."
        )
    if adapter_dir is not None:
        from ml.eval.model_compare import generate_comparison

        baseline, candidate, model_info = generate_comparison(
            records,
            adapter_dir=adapter_dir,
            direction=direction,
            dataset_sha256=data_info["dataset_sha256"],
            device=device,
            batch_size=batch_size,
            base_model_override=base_model_override,
            local_files_only=local_files_only,
            allow_smoke_model=allow_smoke_model,
        )
        warnings.append(model_info["baseline_context_caveat"])
        if model_info["smoke_model"]:
            warnings.append(
                "The adapter is a random smoke model; its scores do not measure "
                "translation quality."
            )
    else:
        assert baseline_predictions is not None and candidate_predictions is not None
        baseline = load_predictions(baseline_predictions, records, field="baseline")
        candidate = load_predictions(candidate_predictions, records, field="candidate")
        model_info = {
            "mode": "precomputed",
            "baseline_predictions_sha256": file_sha256(baseline_predictions),
            "candidate_predictions_sha256": file_sha256(candidate_predictions),
        }
        warnings.append(
            "Prediction provenance and contextual inference cannot be verified in precomputed mode."
        )
    if skip_spbleu:
        warnings.append("spBLEU was explicitly skipped.")
    if skip_comet:
        warnings.append("COMET was explicitly skipped.")
    else:
        warnings.append(
            "Check the COMET model's Sorani coverage; scores may not be calibrated for ckb."
        )
    glossary = load_glossary(glossary_path) if glossary_path is not None else None
    if glossary is None:
        warnings.append("No glossary was supplied; glossary adherence was not computed.")
    comet = (
        load_comet(comet_checkpoint, local_files_only=local_files_only)
        if comet_checkpoint
        else None
    )
    baseline_metrics = compute_metrics(
        records,
        baseline,
        direction=direction,
        glossary=glossary,
        sentencepiece_model=sentencepiece_model,
        comet_model=comet,
        batch_size=batch_size,
        device=device,
    )
    candidate_metrics = compute_metrics(
        records,
        candidate,
        direction=direction,
        glossary=glossary,
        sentencepiece_model=sentencepiece_model,
        comet_model=comet,
        batch_size=batch_size,
        device=device,
    )
    delta = candidate_metrics["chrfpp"] - baseline_metrics["chrfpp"]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "direction": direction,
        "label": label.strip() if label else None,
        "data": data_info,
        "model": model_info,
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "chrfpp_delta": delta,
        "regression": delta < 0,
        "comet_checkpoint_sha256": file_sha256(comet_checkpoint) if comet_checkpoint else None,
        "comet_hparams_sha256": file_sha256(comet_checkpoint.parent.parent / "hparams.yaml")
        if comet_checkpoint
        else None,
        "glossary_sha256": file_sha256(glossary_path) if glossary_path else None,
        "warnings": warnings,
    }
    report, displayed = render_comparison(
        records, baseline, candidate, summary, direction=direction, limit=report_lines
    )
    summary["report_rows"] = displayed
    _write_outputs(target, summary, report, [row.pair_id for row in records], baseline, candidate)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/processed/subtitles/dataset"),
        help="Phase 1 DatasetDict with film-disjoint train/val/test splits",
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("ml/eval/gold_test.jsonl"),
        help="Human-reviewed held-out JSONL references",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/evaluation"),
        help="New or empty directory for metrics, predictions and HTML",
    )
    parser.add_argument("--direction", choices=["en-ckb", "ckb-en"], default="en-ckb")
    parser.add_argument("--adapter-dir", type=Path, help="Completed Phase 2 adapter directory")
    parser.add_argument("--baseline-predictions", type=Path, help="Precomputed baseline JSONL")
    parser.add_argument("--candidate-predictions", type=Path, help="Precomputed candidate JSONL")
    parser.add_argument("--spm-model", type=Path, help="Local SentencePiece model for spBLEU")
    parser.add_argument("--comet-checkpoint", type=Path, help="Trusted reference-based COMET .ckpt")
    parser.add_argument("--skip-spbleu", action="store_true", help="Explicitly skip spBLEU")
    parser.add_argument("--skip-comet", action="store_true", help="Explicitly skip COMET")
    parser.add_argument("--glossary", type=Path, help="JSON array of English/Sorani locked terms")
    parser.add_argument(
        "--allow-unverified-gold", action="store_true", help="Allow unreviewed gold for exploration"
    )
    parser.add_argument(
        "--allow-smoke-model", action="store_true", help="Allow a random smoke-test adapter"
    )
    parser.add_argument("--base-model", help="Override a local base with an identical hashed copy")
    parser.add_argument(
        "--local-files-only", action="store_true", help="Do not fetch model or tokenizer assets"
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--report-lines", type=int, default=100, help="Maximum HTML rows (1–1000)")
    parser.add_argument(
        "--label", help="Optional descriptive label for the report (up to 200 characters)"
    )
    args = parser.parse_args(argv)
    try:
        result = evaluate(
            dataset_dir=args.dataset_dir,
            gold_path=args.gold,
            output_dir=args.output_dir,
            direction=args.direction,
            adapter_dir=args.adapter_dir,
            baseline_predictions=args.baseline_predictions,
            candidate_predictions=args.candidate_predictions,
            sentencepiece_model=args.spm_model,
            comet_checkpoint=args.comet_checkpoint,
            skip_spbleu=args.skip_spbleu,
            skip_comet=args.skip_comet,
            glossary_path=args.glossary,
            allow_unverified_gold=args.allow_unverified_gold,
            allow_smoke_model=args.allow_smoke_model,
            base_model_override=args.base_model,
            local_files_only=args.local_files_only,
            device=args.device,
            batch_size=args.batch_size,
            report_lines=args.report_lines,
            label=args.label,
        )
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("direction", "chrfpp_delta", "regression", "report_rows", "warnings")
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if result["regression"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
