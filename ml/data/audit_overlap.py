"""Audit local candidate pairs against held-out subtitle splits, for review only.

This offline tool never downloads data or models, trains, changes a dataset, or
approves rights. Its limited text signals cannot prove independent works or an
uncontaminated benchmark. See docs/overlap-audit.md before using real material.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, TextIO

from datasets import Dataset, DatasetDict, disable_progress_bars, load_from_disk

from ml.data.overlap_matching import (
    NEAR_MIN_CHARS,
    AuditPair,
    OverlapLimitError,
    PairMatch,
    ReferenceIndex,
    WorkBudget,
    canonical_text,
    prepare_query,
)
from ml.data.stage_text_pairs import (
    TextPairStagingConfig,
    TextPairStagingError,
    _demo_inputs,
    _file_info,
    _identity,
    _read_manifest,
    _write_jsonl,
    stage_text_pairs,
)
from ml.train.dataset import dataset_fingerprint, validate_splits

_MAX_REPORT_BYTES = 1024 * 1024
_MAX_JSONL_LINE_BYTES = 4 * 1024 * 1024
_MAX_TEXT_CHARS = 100_000
_MAX_DATASET_ROWS = 200_000
_MAX_DATASET_BYTES = 256 * 1024 * 1024
_MAX_STAGING_BYTES = 2 * 1024 * 1024 * 1024
_MAX_STAGE_ROWS = 1_000_000
_MAX_DETAIL_RECORDS = 50_000
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_SIGNALS = (
    "same_pair_id",
    "english_exact",
    "sorani_exact",
    "bilingual_exact",
    "english_near",
    "sorani_near",
    "bilingual_near",
    "swapped_bilingual_exact",
)
_SCHEMAS = {"sorani-text-pair-staging-v1", "sorani-tmx-pair-staging-v1"}


class OverlapAuditError(ValueError):
    """Invalid or incomplete audit input; a report must not be published."""


@dataclass(frozen=True, slots=True)
class OverlapAuditConfig:
    dataset_dir: Path
    output_dir: Path
    staging_dirs: tuple[Path, ...] = ()
    gold_en_ckb: Path | None = None
    gold_ckb_en: Path | None = None


@dataclass(frozen=True, slots=True)
class StageInput:
    directory: Path
    label: str
    source: dict[str, Any]
    group_id: str
    report_sha256: str
    manifest_sha256: str
    pairs_sha256: str
    quarantine_sha256: str
    staged: int
    quarantined: int
    total: int
    schema: str
    identity_parts: tuple[str, ...]


@dataclass(slots=True)
class Comparison:
    left_rows: int
    right_rows: int
    shared_film_ids: int | None = None
    matched_row_pairs: int = 0
    left_with_signals: set[str] = field(default_factory=set)
    right_with_signals: set[str] = field(default_factory=set)
    signals: Counter[str] = field(default_factory=Counter)

    def add(self, left: AuditPair, right: AuditPair, match: PairMatch) -> None:
        self.matched_row_pairs += 1
        self.left_with_signals.add(left.pair_id)
        self.right_with_signals.add(right.pair_id)
        self.signals.update(match.signals)

    def report(self) -> dict[str, Any]:
        return {
            "left_rows": self.left_rows,
            "right_rows": self.right_rows,
            "shared_film_ids": self.shared_film_ids,
            "matched_row_pairs": self.matched_row_pairs,
            "left_rows_with_signals": len(self.left_with_signals),
            "right_rows_with_signals": len(self.right_with_signals),
            "signals": {name: self.signals[name] for name in _SIGNALS},
        }


@dataclass(slots=True)
class DetailSink:
    handle: TextIO
    total: int = 0
    written: int = 0

    def add(self, item: dict[str, Any]) -> None:
        self.total += 1
        if self.written < _MAX_DETAIL_RECORDS:
            _write_jsonl(self.handle, item)
            self.written += 1


def _json_object(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise OverlapAuditError("A JSON input contains duplicate keys")
        result[key] = value
    return result


def _constant(_value: str) -> None:
    raise OverlapAuditError("A JSON input contains a non-finite number")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise OverlapAuditError("A JSON input contains a non-finite number")
    return number


def _parse_json(raw: bytes, *, label: str, first: bool = False) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8-sig" if first else "utf-8"),
            object_pairs_hook=_json_object,
            parse_constant=_constant,
            parse_float=_finite_float,
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise OverlapAuditError(f"Invalid UTF-8 JSON object in {label}") from exc
    if not isinstance(value, dict):
        raise OverlapAuditError(f"Expected a JSON object in {label}")
    return value


def _regular_file(path: Path, label: str) -> Path:
    path = path.expanduser()
    if path.is_symlink() or not path.is_file():
        raise OverlapAuditError(f"{label} must be an existing regular file, not a symlink")
    return path.resolve()


def _directory(path: Path, label: str) -> Path:
    path = path.expanduser()
    if path.is_symlink() or not path.is_dir():
        raise OverlapAuditError(f"{label} must be an existing directory, not a symlink")
    return path.resolve()


def _paths(config: OverlapAuditConfig) -> tuple[Path, tuple[Path, ...], dict[str, Path], Path]:
    dataset = _directory(config.dataset_dir, "Dataset")
    stages = tuple(_directory(path, "Staging input") for path in config.staging_dirs)
    if len(set(stages)) != len(stages):
        raise OverlapAuditError("Do not supply the same staging directory more than once")
    gold = {
        direction: _regular_file(path, f"{direction} gold file")
        for direction, path in (("en-ckb", config.gold_en_ckb), ("ckb-en", config.gold_ckb_en))
        if path is not None
    }
    supplied_output = Path(config.output_dir).expanduser()
    if supplied_output.is_symlink() or supplied_output.exists():
        raise OverlapAuditError("Output path already exists; choose a new directory")
    output = supplied_output.resolve()
    for input_dir in (dataset, *stages):
        if (
            output == input_dir
            or output.is_relative_to(input_dir)
            or input_dir.is_relative_to(output)
        ):
            raise OverlapAuditError("Output and input directories must not contain one another")
    if any(
        path == output or path.is_relative_to(output) or output.is_relative_to(path)
        for path in gold.values()
    ):
        raise OverlapAuditError("Output must not overlap a gold input file")
    return dataset, stages, gold, output


def _read_report(path: Path) -> tuple[dict[str, Any], str]:
    path = _regular_file(path, "Staging report")
    with path.open("rb") as handle:
        raw = handle.read(_MAX_REPORT_BYTES + 1)
    if len(raw) > _MAX_REPORT_BYTES:
        raise OverlapAuditError("Staging report exceeds its size limit")
    return _parse_json(raw, label="staging report", first=True), hashlib.sha256(raw).hexdigest()


def _sha(value: Any) -> bool:
    return isinstance(value, str) and bool(_HEX_SHA256.fullmatch(value))


def _count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _stage_input(directory: Path, label: str) -> StageInput:
    report_path = _regular_file(directory / "staging-report.json", "Staging report")
    report, report_sha = _read_report(report_path)
    schema = report.get("schema")
    if (
        not isinstance(schema, str)
        or schema not in _SCHEMAS
        or report.get("status") != "staged_for_review"
        or report.get("training_ready") is not False
    ):
        raise OverlapAuditError("Staging report has an unsupported or incomplete schema/status")
    manifest = _regular_file(directory / "source-manifest.json", "Staging source manifest")
    pairs = _regular_file(directory / "pairs.jsonl", "Staged pairs")
    quarantine = _regular_file(directory / "quarantine.jsonl", "Staging quarantine")
    if any(
        os.path.samefile(left, right)
        for left, right in combinations((report_path, manifest, pairs, quarantine), 2)
    ):
        raise OverlapAuditError("Staging report, manifest, pairs and quarantine must be distinct")
    try:
        source, manifest_sha, _ = _read_manifest(manifest)
    except TextPairStagingError as exc:
        raise OverlapAuditError("Invalid source/rights declaration in staging input") from exc
    inputs, outputs, counts = report.get("inputs"), report.get("outputs"), report.get("counts")
    if not all(isinstance(value, dict) for value in (inputs, outputs, counts)):
        raise OverlapAuditError("Staging report is missing input, output or count metadata")
    assert isinstance(inputs, dict) and isinstance(outputs, dict) and isinstance(counts, dict)
    group_id = _identity("text-pair-collection-v1", source["source_id"])
    # JSON comparison distinguishes booleans from integers (unlike Python's
    # True == 1), while ignoring object key order.
    report_source = json.dumps(report.get("source"), sort_keys=True, separators=(",", ":"))
    declared_source = json.dumps(source, sort_keys=True, separators=(",", ":"))
    if (
        report_source != declared_source
        or report.get("source_group_id") != group_id
        or inputs.get("manifest_sha256") != manifest_sha
    ):
        raise OverlapAuditError("Staging source declaration or manifest hash does not match")
    if pairs.stat().st_size + quarantine.stat().st_size > _MAX_STAGING_BYTES:
        raise OverlapAuditError("Staging outputs exceed the audit byte limit")
    pair_info, quarantine_info = _file_info(pairs), _file_info(quarantine)
    if (
        not _sha(outputs.get("pairs_sha256"))
        or not _sha(outputs.get("quarantine_sha256"))
        or outputs["pairs_sha256"] != pair_info.sha256
        or outputs["quarantine_sha256"] != quarantine_info.sha256
    ):
        raise OverlapAuditError("Staging output hashes do not match the completion report")
    staged = counts.get("staged_pairs")
    if schema == "sorani-text-pair-staging-v1":
        total, quarantined = counts.get("input_pairs"), counts.get("quarantined_pairs")
        en, ckb = inputs.get("english"), inputs.get("sorani")
        if not isinstance(en, dict) or not isinstance(ckb, dict):
            raise OverlapAuditError("Staging report is missing paired-line input identities")
        identity_parts = (en.get("sha256"), ckb.get("sha256"))
    else:
        total, quarantined = counts.get("input_units"), counts.get("quarantined_units")
        tmx = inputs.get("tmx")
        if not isinstance(tmx, dict):
            raise OverlapAuditError("Staging report is missing TMX input identity")
        identity_parts = (tmx.get("sha256"),)
    if not all(_count(value) for value in (staged, total, quarantined)) or not all(
        _sha(value) for value in identity_parts
    ):
        raise OverlapAuditError("Staging report contains invalid counts or input hashes")
    assert isinstance(staged, int) and isinstance(total, int) and isinstance(quarantined, int)
    if (
        staged == 0
        or staged > _MAX_STAGE_ROWS
        or total != staged + quarantined
        or staged != pair_info.lines
        or quarantined != quarantine_info.lines
    ):
        raise OverlapAuditError("Staging counts are empty, inconsistent, or exceed the row limit")
    return StageInput(
        directory,
        label,
        source,
        group_id,
        report_sha,
        manifest_sha,
        pair_info.sha256,
        quarantine_info.sha256,
        staged,
        quarantined,
        total,
        schema,
        identity_parts,
    )


def _jsonl_objects(
    path: Path, expected_sha: str, *, label: str, allow_blank: bool
) -> Iterator[tuple[int, dict[str, Any]]]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        number = 0
        while raw := handle.readline(_MAX_JSONL_LINE_BYTES + 1):
            number += 1
            if len(raw) > _MAX_JSONL_LINE_BYTES:
                raise OverlapAuditError(f"A JSONL line in {label} exceeds the size limit")
            digest.update(raw)
            if not raw.strip():
                if allow_blank:
                    continue
                raise OverlapAuditError(f"Blank JSONL line in {label}")
            yield number, _parse_json(raw, label=f"{label} line {number}", first=number == 1)
    if digest.hexdigest() != expected_sha:
        raise OverlapAuditError(f"{label} changed while it was being read")


def _checked_id(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 2048
        or any(unicodedata.category(char) == "Cc" or char in "\u2028\u2029" for char in value)
    ):
        raise OverlapAuditError(f"A record has an invalid {label}")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise OverlapAuditError(f"A record has an invalid {label}") from exc
    return value


def _checked_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > _MAX_TEXT_CHARS:
        raise OverlapAuditError(f"A record has invalid or oversized {label} text")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise OverlapAuditError(f"A record has invalid Unicode in {label} text") from exc
    normalized = canonical_text(value)
    if not normalized or len(normalized) > _MAX_TEXT_CHARS:
        raise OverlapAuditError(f"A record has invalid or oversized normalized {label} text")
    return normalized


def _pair(row: dict[str, Any], *, stage: bool = False) -> AuditPair:
    pair_id = _checked_id(row.get("pair_id"), "pair ID")
    film_id = None if stage else _checked_id(row.get("film_id"), "film ID")
    return AuditPair(
        pair_id=pair_id,
        film_id=film_id,
        source_group_id=row.get("source_group_id") if stage else None,
        english=_checked_text(row.get("english"), "English"),
        sorani=_checked_text(row.get("sorani"), "Sorani"),
    )


def _dataset(
    path: Path, *, allow_empty_train: bool = False
) -> tuple[dict[str, list[dict[str, Any]]], str]:
    try:
        dataset = load_from_disk(str(path))
        if not isinstance(dataset, DatasetDict) or set(dataset) != {"train", "val", "test"}:
            raise OverlapAuditError("Expected exactly train, val and test DatasetDict splits")
        if sum(len(dataset[name]) for name in dataset) > _MAX_DATASET_ROWS:
            raise OverlapAuditError("Dataset exceeds the audit row limit")
        if sum(dataset[name].data.nbytes for name in dataset) > _MAX_DATASET_BYTES:
            raise OverlapAuditError("Dataset exceeds the audit in-memory byte limit")
        splits = {name: [dict(row) for row in dataset[name]] for name in ("train", "val", "test")}
    except OverlapAuditError:
        raise
    except Exception as exc:
        raise OverlapAuditError("Could not load the trusted local subtitle DatasetDict") from exc
    if not splits["val"] or not splits["test"]:
        raise OverlapAuditError("Both held-out val and test splits must be nonempty")
    if not splits["train"] and not allow_empty_train:
        raise OverlapAuditError("An empty train split requires at least one staged candidate")
    # The trainer's validator requires two nonempty splits. For a review-only
    # audit with no timed training rows, validate the two real held-out splits
    # through it without inventing a training record or changing the dataset.
    validation_view = splits
    if not splits["train"]:
        validation_view = {"train": splits["val"], "val": splits["test"], "test": []}
    try:
        validate_splits(validation_view, allow_unverified=True)
    except ValueError as exc:
        # The validator checks every row before checking film intersections.
        # Report shared film IDs below instead of hiding this important finding.
        if not str(exc).startswith("Film leakage detected between "):
            raise OverlapAuditError("Dataset failed existing split or row validation") from exc
    for rows in splits.values():
        for row in rows:
            _pair(row)
            if row.get("source_context"):
                raise OverlapAuditError(
                    "Explicit source_context is not supported by this pair-only audit; "
                    "screen that dataset separately"
                )
    try:
        return splits, dataset_fingerprint(splits)
    except (TypeError, ValueError) as exc:
        raise OverlapAuditError("Dataset cannot be fingerprinted safely") from exc


def _gold(
    path: Path, direction: str, test_rows: dict[str, dict[str, Any]]
) -> tuple[list[AuditPair], dict[str, Any]]:
    info = _file_info(path)
    source_key = "english" if direction == "en-ckb" else "sorani"
    result: list[AuditPair] = []
    seen: set[str] = set()
    unverified = 0
    for _, row in _jsonl_objects(path, info.sha256, label=f"{direction} gold", allow_blank=True):
        pair = _pair(row)
        original = test_rows.get(pair.pair_id)
        if original is None or pair.film_id != original["film_id"] or pair.pair_id in seen:
            raise OverlapAuditError("Gold contains an unknown, duplicate, or wrong-film test ID")
        if row[source_key] != original[source_key]:
            raise OverlapAuditError("Gold source text differs from its held-out dataset pair")
        review = row.get("review_status", "unverified")
        if not isinstance(review, str) or review not in {"verified", "unverified"}:
            raise OverlapAuditError("Gold contains an invalid review status")
        unverified += int(review != "verified")
        seen.add(pair.pair_id)
        result.append(pair)
    if not result:
        raise OverlapAuditError("Gold file contains no test records")
    return result, {
        "direction": direction,
        "sha256": info.sha256,
        "rows": len(result),
        "unverified_rows": unverified,
    }


def _stage_rows(stage: StageInput, seen: set[str]) -> Iterator[AuditPair]:
    previous = count = 0
    for _, row in _jsonl_objects(
        stage.directory / "pairs.jsonl", stage.pairs_sha256, label=stage.label, allow_blank=False
    ):
        ordinal_key = (
            "source_line" if stage.schema == "sorani-text-pair-staging-v1" else "source_unit_index"
        )
        ordinal = row.get(ordinal_key)
        if not _count(ordinal) or ordinal == 0 or ordinal <= previous or ordinal > stage.total:
            raise OverlapAuditError("Staged rows have invalid or unordered source positions")
        assert isinstance(ordinal, int)
        identity_type = "text-pair-v1" if ordinal_key == "source_line" else "tmx-text-pair-v1"
        expected_id = _identity(
            identity_type,
            stage.group_id,
            stage.source["source_revision"],
            *stage.identity_parts,
            ordinal,
        )
        if (
            row.get("record_type") != "text_pair"
            or row.get("pair_id") != expected_id
            or row.get("source_group_id") != stage.group_id
            or row.get("source_id") != stage.source["source_id"]
            or row.get("source_revision") != stage.source["source_revision"]
            or row.get("source_context") != []
            or row.get("original_direction") != "unknown"
            or row.get("language_verified") is not False
            or row.get("alignment_verified") is not False
        ):
            raise OverlapAuditError("Staged row does not match its review-only source identity")
        if expected_id in seen:
            raise OverlapAuditError("Duplicate staged pair ID across supplied inputs")
        seen.add(expected_id)
        previous = ordinal
        count += 1
        yield _pair(row, stage=True)
    if count != stage.staged:
        raise OverlapAuditError("Staged row count changed during the audit")


def _side(record: AuditPair, label: str) -> dict[str, Any]:
    result: dict[str, Any] = {"input": label, "pair_id": record.pair_id}
    if record.film_id is not None:
        result["film_id"] = record.film_id
    if record.source_group_id is not None:
        result["source_group_id"] = record.source_group_id
    return result


def _compare(
    query: AuditPair,
    query_tokens: Any,
    *,
    left_label: str,
    right_label: str,
    name: str,
    index: ReferenceIndex,
    comparison: Comparison,
    details: DetailSink,
    budget: WorkBudget,
) -> None:
    for reference, match in index.find(query_tokens, budget):
        comparison.add(query, reference, match)
        details.add(
            {
                "type": "pair_signal",
                "comparison": name,
                "left": _side(query, left_label),
                "right": _side(reference, right_label),
                "signals": sorted(match.signals),
                "similarities": dict(sorted(match.similarities.items())),
            }
        )


def _publish(stage: Path, output: Path) -> None:
    try:
        output.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise OverlapAuditError("Output path already exists; choose a new directory") from exc
    created: list[Path] = []
    try:
        # The report is the completion marker and is copied last.
        for name in ("matches.jsonl", "overlap-report.json"):
            destination = output / name
            with destination.open("xb") as target:
                created.append(destination)
                with (stage / name).open("rb") as source:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
    except BaseException:
        for path in reversed(created):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            output.rmdir()
        except OSError:
            pass
        raise


def audit_overlap(config: OverlapAuditConfig) -> dict[str, Any]:
    """Audit explicitly supplied local inputs; publish a private, non-approval report."""
    dataset_dir, stage_dirs, gold_paths, output = _paths(config)
    splits, fingerprint = _dataset(dataset_dir, allow_empty_train=bool(stage_dirs))
    stage_inputs = [
        _stage_input(path, f"staging-{index}") for index, path in enumerate(stage_dirs, 1)
    ]
    if sum(stage.staged for stage in stage_inputs) > _MAX_STAGE_ROWS:
        raise OverlapAuditError("Combined staged inputs exceed the audit row limit")
    dataset_pairs = {
        name: sorted((_pair(row) for row in rows), key=lambda pair: pair.pair_id)
        for name, rows in splits.items()
    }
    test_rows = {row["pair_id"]: row for row in splits["test"]}
    gold_pairs: dict[str, list[AuditPair]] = {}
    gold_reports: list[dict[str, Any]] = []
    for direction, path in gold_paths.items():
        label = f"test-gold-{direction}"
        gold_pairs[label], gold_report = _gold(path, direction, test_rows)
        gold_reports.append(gold_report)

    budget = WorkBudget()
    references = {"val": dataset_pairs["val"], "test": dataset_pairs["test"], **gold_pairs}
    indexes = {name: ReferenceIndex(rows, budget) for name, rows in references.items()}
    comparisons: dict[str, Comparison] = {}
    group_pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    film_sets = {name: {pair.film_id for pair in rows} for name, rows in dataset_pairs.items()}
    for left, right in group_pairs:
        comparisons[f"{left}_vs_{right}"] = Comparison(
            len(dataset_pairs[left]),
            len(dataset_pairs[right]),
            len(film_sets[left] & film_sets[right]),
        )
    for gold_name, rows in gold_pairs.items():
        for left in ("train", "val"):
            comparisons[f"{left}_vs_{gold_name}"] = Comparison(len(dataset_pairs[left]), len(rows))
    for stage in stage_inputs:
        for name, rows in references.items():
            comparisons[f"{stage.label}_vs_{name}"] = Comparison(stage.staged, len(rows))

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.audit-", dir=output.parent) as temp:
        temporary = Path(temp)
        with (temporary / "matches.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            details = DetailSink(handle)
            for left, right in group_pairs:
                name = f"{left}_vs_{right}"
                for film_id in sorted(film_sets[left] & film_sets[right]):
                    details.add({"type": "shared_film_id", "comparison": name, "film_id": film_id})
            for left, targets in (
                ("train", ["val", "test", *gold_pairs]),
                ("val", ["test", *gold_pairs]),
            ):
                for pair in dataset_pairs[left]:
                    query = prepare_query(pair, budget)
                    for right in targets:
                        name = f"{left}_vs_{right}"
                        _compare(
                            pair,
                            query,
                            left_label=left,
                            right_label=right,
                            name=name,
                            index=indexes[right],
                            comparison=comparisons[name],
                            details=details,
                            budget=budget,
                        )
            seen_stage_ids: set[str] = set()
            for stage in stage_inputs:
                for pair in _stage_rows(stage, seen_stage_ids):
                    query = prepare_query(pair, budget)
                    for right in references:
                        name = f"{stage.label}_vs_{right}"
                        _compare(
                            pair,
                            query,
                            left_label=stage.label,
                            right_label=right,
                            name=name,
                            index=indexes[right],
                            comparison=comparisons[name],
                            details=details,
                            budget=budget,
                        )
        # Re-read before publication. Hashes describe the data actually read,
        # but are not signatures and cannot defeat a malicious concurrent writer.
        _, after_fingerprint = _dataset(dataset_dir, allow_empty_train=bool(stage_dirs))
        if fingerprint != after_fingerprint:
            raise OverlapAuditError("Dataset changed during the overlap audit")
        for direction, path in gold_paths.items():
            expected = next(
                item["sha256"] for item in gold_reports if item["direction"] == direction
            )
            if _file_info(path).sha256 != expected:
                raise OverlapAuditError("A gold input changed during the overlap audit")
        for stage in stage_inputs:
            hashes = {
                "staging-report.json": stage.report_sha256,
                "source-manifest.json": stage.manifest_sha256,
                "pairs.jsonl": stage.pairs_sha256,
                "quarantine.jsonl": stage.quarantine_sha256,
            }
            if any(
                _file_info(stage.directory / name).sha256 != value for name, value in hashes.items()
            ):
                raise OverlapAuditError("A staging input changed during the overlap audit")
        detail_info = _file_info(temporary / "matches.jsonl")
        any_findings = any(
            comparison.matched_row_pairs or comparison.shared_film_ids
            for comparison in comparisons.values()
        )
        report = {
            "schema": "sorani-pair-overlap-audit-v1",
            "status": "review_needed" if any_findings else "no_signals_detected",
            "training_ready": False,
            "dataset": {
                "training_fingerprint_sha256": fingerprint,
                "split_rows": {name: len(rows) for name, rows in dataset_pairs.items()},
                "empty_train_split": not bool(dataset_pairs["train"]),
            },
            "staging_inputs": [
                {
                    "label": stage.label,
                    "schema": stage.schema,
                    "source_id": stage.source["source_id"],
                    "source_revision": stage.source["source_revision"],
                    "source_group_id": stage.group_id,
                    "declared_rights_status": stage.source["rights"]["status"],
                    "declared_permitted_uses": stage.source["rights"]["permitted_uses"],
                    "staging_report_sha256": stage.report_sha256,
                    "source_manifest_sha256": stage.manifest_sha256,
                    "pairs_sha256": stage.pairs_sha256,
                    "quarantine_sha256": stage.quarantine_sha256,
                    "staged_pairs": stage.staged,
                    "quarantined_not_compared": stage.quarantined,
                }
                for stage in stage_inputs
            ],
            "gold_inputs": gold_reports,
            "gold_not_supplied": [
                direction for direction in ("en-ckb", "ckb-en") if direction not in gold_paths
            ],
            "method": {
                "normalization": "Unicode NFKC, casefold, whitespace collapse; ZWNJ retained",
                "near_metric": "distinct character-trigram Jaccard >= 0.90 on each matching side",
                "near_min_normalized_characters_per_side": NEAR_MIN_CHARS,
                "near_retrieval": (
                    "deterministic prefix index followed by exact Jaccard verification"
                ),
                "group_comparison": "exact dataset film_id equality only",
            },
            "comparisons": {name: comparison.report() for name, comparison in comparisons.items()},
            "details": {
                "filename": "matches.jsonl",
                "sha256": detail_info.sha256,
                "total_records": details.total,
                "written_records": details.written,
                "truncated": details.written != details.total,
            },
            "work": {
                "reference_windows": budget.reference_windows,
                "query_windows": budget.query_windows,
                "posting_visits": budget.posting_visits,
                "candidate_checks": budget.candidate_checks,
            },
            "limitations": [
                "Signals are review leads, not proof of leakage or independence; "
                "short phrases may be common.",
                "Only supplied retained pair texts and optional gold variants are compared; "
                "quarantined, filtered, raw, and unsupplied data are not checked.",
                "Near matches below 12 normalized characters are not checked; "
                "exact matches remain enabled.",
                "Paraphrases, different segmentation, language-specific spelling, "
                "reconstructed contexts, and base-model pretraining can still overlap.",
                "Different film IDs do not establish independent works; "
                "source groups are not comparable to film IDs.",
                "No timing, language, translation quality, rights, provenance, "
                "or commercial use is verified.",
                "The audit is not connected to the trainer or web app "
                "and does not make staged data trainer-ready.",
            ],
        }
        (temporary / "overlap-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        _publish(temporary, output)
    return report


def _demo_dataset(directory: Path) -> Path:
    """Make mock split rows only; their IDs and start values are synthetic."""
    disable_progress_bars()

    def row(film: str, number: int, english: str, sorani: str) -> dict[str, Any]:
        return {
            "pair_id": f"synthetic-{film}-{number}",
            "film_id": f"synthetic-{film}",
            "english": english,
            "sorani": sorani,
            "english_start_ms": number * 1000,
            "sorani_start_ms": number * 1000,
            "provenance": "unknown",
            "language_verified": False,
        }

    dataset = DatasetDict(
        {
            "train": Dataset.from_list(
                [row("train", 1, "Please open the window.", "تکایە پەنجەرەکە بکەوە.")]
            ),
            "val": Dataset.from_list(
                [row("val", 1, "Keep the lantern lit.", "چراغەکە ڕوون بهێڵەوە.")]
            ),
            "test": Dataset.from_list(
                [
                    row("test", 1, "PLEASE  OPEN THE WINDOW.", "تکایە پەنجەرەکە بکەوە."),
                    row("test", 2, "The garden needs water!", "باخچەکە پێویستی بە ئاوە."),
                ]
            ),
        }
    )
    path = directory / "synthetic-dataset"
    dataset.save_to_disk(str(path))
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Use synthetic mock inputs only")
    parser.add_argument("--dataset-dir", type=Path, help="Local timed-subtitle DatasetDict")
    parser.add_argument(
        "--staging-dir",
        type=Path,
        action="append",
        default=[],
        help="Completed review-only text/TMX staging directory; may be repeated",
    )
    parser.add_argument("--gold-en-ckb", type=Path, help="Optional EN→CKB test gold JSONL")
    parser.add_argument("--gold-ckb-en", type=Path, help="Optional CKB→EN test gold JSONL")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New private output directory"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.demo and (args.dataset_dir or args.staging_dir or args.gold_en_ckb or args.gold_ckb_en):
        parser.error("--demo cannot be combined with dataset, staging, or gold inputs")
    if not args.demo and args.dataset_dir is None:
        parser.error("supply --dataset-dir, or use --demo")
    try:
        if args.demo:
            with tempfile.TemporaryDirectory(prefix="sorani-synthetic-overlap-") as temp:
                root = Path(temp)
                english, sorani, manifest = _demo_inputs(root)
                staging_dir = root / "synthetic-stage"
                stage_text_pairs(TextPairStagingConfig(english, sorani, manifest, staging_dir))
                report = audit_overlap(
                    OverlapAuditConfig(_demo_dataset(root), args.output_dir, (staging_dir,))
                )
        else:
            report = audit_overlap(
                OverlapAuditConfig(
                    args.dataset_dir,
                    args.output_dir,
                    tuple(args.staging_dir),
                    args.gold_en_ckb,
                    args.gold_ckb_en,
                )
            )
    except (OverlapAuditError, OverlapLimitError, TextPairStagingError, OSError) as exc:
        print(f"Overlap audit failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "status": report["status"],
                "matched_row_pairs": sum(
                    c["matched_row_pairs"] for c in report["comparisons"].values()
                ),
                "shared_film_ids": sum(
                    c["shared_film_ids"] or 0 for c in report["comparisons"].values()
                ),
                "details_truncated": report["details"]["truncated"],
                "training_ready": False,
                "note": (
                    "Review signals only; absence of a match does not establish "
                    "independence or rights."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if report["status"] == "review_needed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
