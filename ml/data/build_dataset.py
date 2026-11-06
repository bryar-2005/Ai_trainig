"""Build a film-disjoint Hugging Face DatasetDict from local subtitles.

Example: python -m ml.data.build_dataset --input-dir data/raw --output-dir
 data/processed/subtitles --lid-model data/cache/lid.176.ftz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, Features, Sequence, Value, disable_progress_bars

from ml.data.adapters.local import FilmFiles, LocalDataError, discover_films
from ml.data.align import AlignedPair, align_cues
from ml.data.filter_dedupe import FastTextLanguageIdentifier, deduplicate_pairs, filter_pair
from ml.data.loaders import Cue, SubtitleLoadError, load_subtitles
from ml.data.normalize import HehPolicy, normalization_warnings, normalize_english, normalize_sorani
from ml.data.split import sample_gold_for_review, split_by_film
from ml.data.stats import dataset_statistics, markdown_report

_OUTPUT_MARKER = ".sorani-dataset-output"
FEATURES = Features(
    {
        "pair_id": Value("string"),
        "film_id": Value("string"),
        "english": Value("string"),
        "sorani": Value("string"),
        "english_cue_ids": Sequence(Value("int32")),
        "sorani_cue_ids": Sequence(Value("int32")),
        "english_cue_starts_ms": Sequence(Value("int64")),
        "english_cue_ends_ms": Sequence(Value("int64")),
        "sorani_cue_starts_ms": Sequence(Value("int64")),
        "sorani_cue_ends_ms": Sequence(Value("int64")),
        "english_cue_texts": Sequence(Value("string")),
        "sorani_cue_texts": Sequence(Value("string")),
        "english_cue_speaker_labels": Sequence(Sequence(Value("string"))),
        "sorani_cue_speaker_labels": Sequence(Sequence(Value("string"))),
        "english_cue_sound_effects": Sequence(Sequence(Value("string"))),
        "sorani_cue_sound_effects": Sequence(Sequence(Value("string"))),
        "english_cue_styles": Sequence(Value("string")),
        "sorani_cue_styles": Sequence(Value("string")),
        "english_start_ms": Value("int64"),
        "english_end_ms": Value("int64"),
        "sorani_start_ms": Value("int64"),
        "sorani_end_ms": Value("int64"),
        "english_speakers": Sequence(Value("string")),
        "sorani_speakers": Sequence(Value("string")),
        "english_sound_effects": Sequence(Value("string")),
        "sorani_sound_effects": Sequence(Value("string")),
        "english_styles": Sequence(Value("string")),
        "sorani_styles": Sequence(Value("string")),
        "temporal_iou": Value("float64"),
        "length_ratio": Value("float64"),
        "sorani_script_ratio": Value("float64"),
        "language_verified": Value("bool"),
        "english_lid_label": Value("string"),
        "english_lid_confidence": Value("float64"),
        "sorani_lid_label": Value("string"),
        "sorani_lid_confidence": Value("float64"),
        "normalization_warnings": Sequence(Value("string")),
        "provenance": Value("string"),
        "quality_score": Value("float64"),
        "quality_source": Value("string"),
        "source_license": Value("string"),
        "attribution": Value("string"),
    }
)


@dataclass(frozen=True, slots=True)
class BuildConfig:
    input_dir: Path
    output_dir: Path
    gold_path: Path
    lid_model: Path | None = None
    skip_language_id: bool = False
    lid_threshold: float = 0.3
    min_iou: float = 0.5
    tolerance_ms: int = 500
    max_group_size: int = 3
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    gold_size: int = 1000
    seed: int = 42
    heh_policy: HehPolicy = "preserve"
    encoding: str = "utf-8-sig"
    allow_empty_splits: bool = False
    overwrite: bool = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _unique(values: list[str | None]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _joined(cues: tuple[Cue, ...]) -> str:
    return " ".join(" ".join(cue.text.split()) for cue in cues if cue.text.strip())


def _pair_record(
    film: FilmFiles,
    alignment: AlignedPair,
    *,
    heh_policy: HehPolicy,
    detector: FastTextLanguageIdentifier | None,
) -> tuple[dict[str, Any] | None, str | None]:
    en_cues, ckb_cues = alignment.english_cues, alignment.sorani_cues
    english = normalize_english(_joined(en_cues))
    sorani = normalize_sorani(_joined(ckb_cues), heh_policy=heh_policy)
    decision = filter_pair(english, sorani, detector=detector)
    if not decision.accepted:
        return None, decision.reason
    identity = json.dumps(
        [film.film_id, [cue.index for cue in en_cues], [cue.index for cue in ckb_cues]],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    en_language, ckb_language = decision.english_language, decision.sorani_language
    return (
        {
            "pair_id": hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24],
            "film_id": film.film_id,
            "english": english,
            "sorani": sorani,
            "english_cue_ids": [cue.index for cue in en_cues],
            "sorani_cue_ids": [cue.index for cue in ckb_cues],
            "english_cue_starts_ms": [cue.start_ms for cue in en_cues],
            "english_cue_ends_ms": [cue.end_ms for cue in en_cues],
            "sorani_cue_starts_ms": [cue.start_ms for cue in ckb_cues],
            "sorani_cue_ends_ms": [cue.end_ms for cue in ckb_cues],
            "english_cue_texts": [normalize_english(cue.text) for cue in en_cues],
            "sorani_cue_texts": [normalize_sorani(cue.text, heh_policy) for cue in ckb_cues],
            "english_cue_speaker_labels": [list(cue.speaker_labels) for cue in en_cues],
            "sorani_cue_speaker_labels": [list(cue.speaker_labels) for cue in ckb_cues],
            "english_cue_sound_effects": [list(cue.sound_effects) for cue in en_cues],
            "sorani_cue_sound_effects": [list(cue.sound_effects) for cue in ckb_cues],
            "english_cue_styles": [cue.style for cue in en_cues],
            "sorani_cue_styles": [cue.style for cue in ckb_cues],
            "english_start_ms": min(cue.start_ms for cue in en_cues),
            "english_end_ms": max(cue.end_ms for cue in en_cues),
            "sorani_start_ms": min(cue.start_ms for cue in ckb_cues),
            "sorani_end_ms": max(cue.end_ms for cue in ckb_cues),
            "english_speakers": _unique(
                [speaker for cue in en_cues for speaker in cue.speaker_labels]
            ),
            "sorani_speakers": _unique(
                [speaker for cue in ckb_cues for speaker in cue.speaker_labels]
            ),
            "english_sound_effects": [effect for cue in en_cues for effect in cue.sound_effects],
            "sorani_sound_effects": [effect for cue in ckb_cues for effect in cue.sound_effects],
            "english_styles": _unique([cue.style for cue in en_cues]),
            "sorani_styles": _unique([cue.style for cue in ckb_cues]),
            "temporal_iou": alignment.temporal_iou,
            "length_ratio": decision.length_ratio,
            "sorani_script_ratio": decision.script_ratio,
            "language_verified": detector is not None,
            "english_lid_label": en_language.label if en_language else None,
            "english_lid_confidence": en_language.confidence if en_language else None,
            "sorani_lid_label": ckb_language.label if ckb_language else None,
            "sorani_lid_confidence": ckb_language.confidence if ckb_language else None,
            "normalization_warnings": normalization_warnings(sorani),
            "provenance": film.provenance,
            # A temporal fallback is a provenance signal, NOT semantic quality.
            "quality_score": film.quality_score
            if film.quality_score is not None
            else alignment.temporal_iou,
            "quality_source": "metadata" if film.quality_score is not None else "temporal_iou",
            "source_license": film.source_license,
            "attribution": film.attribution,
        },
        None,
    )


def _has_output_marker(output: Path) -> bool:
    marker = output / _OUTPUT_MARKER
    if marker.is_symlink() or not marker.is_file():
        return False
    try:
        return marker.read_text(encoding="utf-8") == "sorani-subtitle-dataset-v1\n"
    except (OSError, UnicodeError):
        return False


def _validated_paths(config: BuildConfig) -> tuple[Path, Path, Path]:
    if config.input_dir.is_symlink():
        raise ValueError("The input directory itself may not be a symlink")
    if config.output_dir.is_symlink() or config.gold_path.is_symlink():
        raise ValueError("Output and gold paths may not be symlinks")
    root, output, gold = (
        config.input_dir.expanduser().resolve(),
        config.output_dir.expanduser().resolve(),
        config.gold_path.expanduser().resolve(),
    )
    if root == output or root.is_relative_to(output) or output.is_relative_to(root):
        raise ValueError("Input and output directories must not contain one another")
    if gold == output or gold.is_relative_to(output) or gold.is_relative_to(root):
        raise ValueError("Gold output must be outside both the input and output directories")
    if output.exists():
        if not output.is_dir():
            raise ValueError(f"Output path is not a directory: {output}")
        if not config.overwrite:
            raise ValueError(f"Output already exists: {output}; pass --overwrite to replace it")
        if not _has_output_marker(output):
            raise ValueError(f"Refusing to overwrite an unmarked directory: {output}")
    if gold.exists() and (gold.is_dir() or not config.overwrite):
        raise ValueError(
            f"Gold path already exists or is a directory: {gold}; use --overwrite for a file"
        )
    return root, output, gold


def _write_outputs(
    output: Path,
    gold_path: Path,
    splits: dict[str, list[dict[str, Any]]],
    gold_rows: list[dict[str, Any]],
    stats: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    disable_progress_bars()
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.build-", dir=output.parent) as temp:
        stage = Path(temp) / "output"
        stage.mkdir()
        dataset = DatasetDict(
            {
                name: (
                    Dataset.from_list(rows, features=FEATURES)
                    if rows
                    else Dataset.from_dict({column: [] for column in FEATURES}, features=FEATURES)
                )
                for name, rows in splits.items()
            }
        )
        # datasets may otherwise write zero Arrow shards for empty splits, which
        # makes those splits impossible to reload. Nonempty splits retain
        # automatic shard sizing.
        empty_split_shards = {name: 1 for name, rows in splits.items() if not rows}
        dataset.save_to_disk(str(stage / "dataset"), num_shards=empty_split_shards)
        (stage / "stats.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (stage / "stats.md").write_text(markdown_report(stats), encoding="utf-8")
        (stage / _OUTPUT_MARKER).write_text("sorani-subtitle-dataset-v1\n", encoding="utf-8")
        gold_temp: Path | None = None
        backup = Path(temp) / "previous-output"
        installed = False
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=gold_path.parent, prefix=".gold-", delete=False
            ) as handle:
                gold_temp = Path(handle.name)
                for row in gold_rows:
                    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            # Recheck before replacing any existing path. Only replace outputs we created.
            if output.is_symlink() or (
                output.exists()
                and (not overwrite or not output.is_dir() or not _has_output_marker(output))
            ):
                raise ValueError(f"Output changed or is not safe to overwrite: {output}")
            if gold_path.is_symlink() or (
                gold_path.exists() and (not overwrite or gold_path.is_dir())
            ):
                raise ValueError(f"Gold path changed or is not safe to overwrite: {gold_path}")
            if output.exists():
                output.rename(backup)
            stage.rename(output)
            installed = True
            os.replace(gold_temp, gold_path)
            gold_temp = None
        except BaseException:
            if installed:
                shutil.rmtree(output)
            if backup.exists():
                backup.rename(output)
            raise
        finally:
            if gold_temp is not None:
                gold_temp.unlink(missing_ok=True)


def build_dataset(config: BuildConfig) -> dict[str, Any]:
    """Build and atomically install output; returns the JSON-compatible report."""
    if (config.lid_model is None) == (not config.skip_language_id):
        raise ValueError("Choose exactly one: --lid-model or --skip-language-id")
    if not 0.5 <= config.min_iou <= 1:
        raise ValueError("Minimum IoU must be between 0.5 and 1")
    if not 0 <= config.tolerance_ms <= 500:
        raise ValueError("Timestamp tolerance must be between 0 and 500 ms")
    if not 1 <= config.max_group_size <= 10:
        raise ValueError("Maximum group size must be between 1 and 10")
    if config.gold_size < 0:
        raise ValueError("Gold sample size must be non-negative")
    root, output, gold = _validated_paths(config)
    films = discover_films(root)
    detector = (
        FastTextLanguageIdentifier(config.lid_model, min_confidence=config.lid_threshold)
        if config.lid_model is not None
        else None
    )
    counts: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for film in films:
        english = load_subtitles(film.english_path, encoding=config.encoding)
        sorani = load_subtitles(film.sorani_path, encoding=config.encoding)
        counts.update({"english_cues": len(english), "sorani_cues": len(sorani)})
        counts["removed_ad_lines"] += sum(cue.removed_ad_lines for cue in (*english, *sorani))
        english_dialogue = [cue for cue in english if cue.text.strip()]
        sorani_dialogue = [cue for cue in sorani if cue.text.strip()]
        counts["english_non_dialogue_cues"] += len(english) - len(english_dialogue)
        counts["sorani_non_dialogue_cues"] += len(sorani) - len(sorani_dialogue)
        alignments = align_cues(
            english_dialogue,
            sorani_dialogue,
            min_iou=config.min_iou,
            tolerance_ms=config.tolerance_ms,
            max_group_size=config.max_group_size,
        )
        counts["aligned_pairs"] += len(alignments)
        counts["english_unmatched_cues"] += len(english_dialogue) - sum(
            len(pair.english_cues) for pair in alignments
        )
        counts["sorani_unmatched_cues"] += len(sorani_dialogue) - sum(
            len(pair.sorani_cues) for pair in alignments
        )
        for alignment in alignments:
            record, reason = _pair_record(
                film, alignment, heh_policy=config.heh_policy, detector=detector
            )
            if record is not None:
                records.append(record)
            else:
                rejected[reason or "unknown"] += 1
        inputs.append(
            {
                "film_id": film.film_id,
                "english_file": film.english_path.name,
                "english_sha256": _sha256(film.english_path),
                "sorani_file": film.sorani_path.name,
                "sorani_sha256": _sha256(film.sorani_path),
                "metadata_sha256": (
                    _sha256(film.english_path.parent / "metadata.json")
                    if (film.english_path.parent / "metadata.json").is_file()
                    else None
                ),
                "provenance": film.provenance,
                "quality_score": film.quality_score,
                "source_license": film.source_license,
                "attribution": film.attribution,
            }
        )
    retained, duplicate_count = deduplicate_pairs(records)
    if not retained:
        raise ValueError(
            "No subtitle pairs survived alignment and filtering; "
            f"aligned_pairs={counts['aligned_pairs']}, "
            f"filter_rejections={dict(sorted(rejected.items()))}. "
            "Check timestamps, text, and language-ID settings."
        )
    splits = split_by_film(
        retained,
        val_fraction=config.val_fraction,
        test_fraction=config.test_fraction,
        seed=config.seed,
        allow_empty_splits=config.allow_empty_splits,
    )
    gold_rows = sample_gold_for_review(splits["test"], size=config.gold_size, seed=config.seed)
    warnings = []
    if detector is None:
        warnings.append(
            "Language identification was skipped: Arabic script alone does not verify Sorani."
        )
    if config.heh_policy == "positional":
        warnings.append(
            "Positional ه normalization is lossy; a Sorani speaker must review the output."
        )
    elif any(row["normalization_warnings"] for row in retained):
        warnings.append(
            "Ambiguous Arabic ه was preserved and flagged, not automatically disambiguated."
        )
    if len(gold_rows) < config.gold_size:
        warnings.append(
            f"The held-out test split has only {len(gold_rows)} pairs for the "
            f"{config.gold_size}-pair review sample; no examples were duplicated."
        )
    if any(not splits[name] for name in ("train", "val", "test")):
        warnings.append(
            "At least one split is empty; this dataset is not suitable for a full benchmark."
        )
    missing_licenses = sum(not film.source_license for film in films)
    if missing_licenses:
        warnings.append(
            f"License metadata was not supplied for {missing_licenses} input films; verify rights."
        )
    diagnostics = {
        "input_film_count": len(films),
        "cue_counts": dict(sorted(counts.items())),
        "filter_rejections": dict(sorted(rejected.items())),
        "duplicates_removed": duplicate_count,
        "deduplication": "bilingual char-trigram MinHash LSH; exact per-language Jaccard >= 0.90",
        "deduplication_note": (
            "LSH near-duplicate candidate retrieval is probabilistic; "
            "exact pairs are always removed."
        ),
        "language_id": "fasttext" if detector else "unverified",
        "lid_model_sha256": _sha256(config.lid_model) if config.lid_model else None,
        "lid_confidence_threshold": config.lid_threshold if detector else None,
        "heh_policy": config.heh_policy,
        "input_encoding": config.encoding,
        "min_iou": config.min_iou,
        "tolerance_ms": config.tolerance_ms,
        "max_group_size": config.max_group_size,
        "val_fraction": config.val_fraction,
        "test_fraction": config.test_fraction,
        "seed": config.seed,
        "inputs": inputs,
    }
    stats = dataset_statistics(
        splits,
        diagnostics=diagnostics,
        gold_count=len(gold_rows),
        gold_requested=config.gold_size,
        warnings=warnings,
    )
    _write_outputs(output, gold, splits, gold_rows, stats, overwrite=config.overwrite)
    return stats


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, required=True, help="Local directory of film folders"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New directory for DatasetDict and reports"
    )
    parser.add_argument("--gold-path", type=Path, default=Path("ml/eval/gold_test.jsonl"))
    language = parser.add_mutually_exclusive_group(required=True)
    language.add_argument(
        "--lid-model", type=Path, help="Local fastText model with en and ckb labels"
    )
    language.add_argument(
        "--skip-language-id",
        action="store_true",
        help="Explicitly mark results as language-unverified",
    )
    parser.add_argument("--lid-threshold", type=float, default=0.3)
    parser.add_argument("--min-iou", type=float, default=0.5)
    parser.add_argument("--tolerance-ms", type=int, default=500)
    parser.add_argument("--max-group-size", type=int, default=3)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--gold-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--heh-policy", choices=["preserve", "positional"], default="preserve")
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument(
        "--allow-empty-splits", action="store_true", help="Development only; no leakage"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace only marked output and gold files"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        stats = build_dataset(BuildConfig(**vars(args)))
    except (LocalDataError, SubtitleLoadError, ValueError, OSError) as exc:
        print(f"Dataset build failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "pair_count": stats["pair_count"],
                "film_count": stats["film_count"],
                "split_counts": {
                    name: part["pair_count"] for name, part in stats["splits"].items()
                },
                "gold_written": stats["gold_review_sample"]["written"],
                "output_dir": str(args.output_dir),
                "gold_path": str(args.gold_path),
                "warnings": stats["warnings"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
