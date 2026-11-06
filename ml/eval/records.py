"""Load reviewed test pairs with film-scoped context and original timings."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_from_disk

from ml.train.dataset import (
    ContextExample,
    dataset_fingerprint,
    make_context_examples,
    validate_splits,
)


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    pair_id: str
    film_id: str
    source: str
    reference: str
    duration_ms: int
    provenance: str
    example: ContextExample


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_evaluation_records(
    dataset_dir: Path, gold_path: Path, *, direction: str, allow_unverified_gold: bool = False
) -> tuple[list[EvaluationRecord], dict[str, Any]]:
    """Use the complete test film for context, but score only reviewed gold IDs.

    The source text and film identity must still match the held-out DatasetDict;
    the reviewed target text may be corrected in the gold JSONL file.
    """
    if direction not in {"en-ckb", "ckb-en"}:
        raise ValueError("direction must be en-ckb or ckb-en")
    if not dataset_dir.is_dir():
        raise ValueError(f"Dataset directory does not exist: {dataset_dir}")
    if not gold_path.is_file():
        raise ValueError(f"Gold review file does not exist: {gold_path}")
    dataset = load_from_disk(str(dataset_dir))
    if not isinstance(dataset, DatasetDict) or not {"train", "val", "test"}.issubset(dataset):
        raise ValueError("Expected train, val and test splits in the Phase 1 DatasetDict")
    splits = {name: [dict(row) for row in dataset[name]] for name in ("train", "val", "test")}
    validate_splits(splits, allow_unverified=True)
    if not splits["test"]:
        raise ValueError("The held-out test split is empty")
    source_key, target_key = (
        ("english", "sorani") if direction == "en-ckb" else ("sorani", "english")
    )
    test_rows = {row["pair_id"]: row for row in splits["test"]}
    contexts = {
        item.pair_id: item
        for item in make_context_examples(splits["test"], direction=direction, context_sentences=5)
    }
    try:
        raw_lines = gold_path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Cannot read gold file: {exc}") from exc
    records: list[EvaluationRecord] = []
    seen: set[str] = set()
    unverified = 0
    for line_number, line in enumerate(raw_lines, start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on gold line {line_number}: {exc.msg}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"Gold line {line_number} must be a JSON object")
        pair_id = item.get("pair_id")
        if not isinstance(pair_id, str) or pair_id not in test_rows:
            raise ValueError(f"Gold line {line_number} does not identify a held-out test pair")
        if pair_id in seen:
            raise ValueError(f"Duplicate gold pair_id: {pair_id}")
        seen.add(pair_id)
        original = test_rows[pair_id]
        if item.get("film_id") != original["film_id"]:
            raise ValueError(f"Gold film ID differs from the held-out pair {pair_id}")
        for key in (source_key, target_key):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError(f"Gold pair {pair_id} has no {key} text")
        if item[source_key] != original[source_key]:
            raise ValueError(f"Gold source text differs from the held-out pair {pair_id}")
        review = item.get("review_status", "unverified")
        if not isinstance(review, str) or review not in {"verified", "unverified"}:
            raise ValueError(f"Gold pair {pair_id} has an invalid review_status")
        if review != "verified":
            unverified += 1
            if not allow_unverified_gold:
                raise ValueError(
                    "Gold contains unverified rows; review them first or explicitly pass "
                    "--allow-unverified-gold for exploratory results"
                )
        start, end = original.get(f"{source_key}_start_ms"), original.get(f"{source_key}_end_ms")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end <= start
        ):
            raise ValueError(f"Test pair {pair_id} lacks a positive original source duration")
        reference = item[target_key].strip()
        records.append(
            EvaluationRecord(
                pair_id=pair_id,
                film_id=original["film_id"],
                source=original[source_key],
                reference=reference,
                duration_ms=end - start,
                provenance=original["provenance"],
                example=replace(contexts[pair_id], target=reference),
            )
        )
    if not records:
        raise ValueError("The gold review file contains no evaluation records")
    # The training fingerprint deliberately omits end times, which do affect
    # CPS. Record an additional digest so timing changes cannot go unnoticed.
    timing = [
        [
            row["pair_id"],
            row.get("english_start_ms"),
            row.get("english_end_ms"),
            row.get("sorani_start_ms"),
            row.get("sorani_end_ms"),
        ]
        for row in sorted(splits["test"], key=lambda row: row["pair_id"])
    ]
    timing_sha256 = hashlib.sha256(
        json.dumps(timing, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return records, {
        "dataset_sha256": dataset_fingerprint(splits),
        "test_timing_sha256": timing_sha256,
        "gold_sha256": file_sha256(gold_path),
        "unverified_gold_count": unverified,
        "language_unverified_test_pairs": sum(
            not row["language_verified"] for row in splits["test"]
        ),
        "test_pairs": len(splits["test"]),
        "scored_pairs": len(records),
    }


def load_predictions(
    path: Path, records: list[EvaluationRecord], *, field: str = "text"
) -> list[str]:
    """Read pair-id keyed text or a baseline/candidate field from an earlier report."""
    if not path.is_file():
        raise ValueError(f"Prediction file does not exist: {path}")
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Cannot read predictions: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid prediction JSON on line {line_number}") from exc
        if not isinstance(item, dict) or not isinstance(item.get("pair_id"), str):
            raise ValueError(f"Prediction line {line_number} must have a pair_id")
        text = item.get("text", item.get(field))
        if not isinstance(text, str):
            raise ValueError(f"Prediction line {line_number} must have string text or {field}")
        if item["pair_id"] in values:
            raise ValueError(f"Duplicate prediction for {item['pair_id']}")
        values[item["pair_id"]] = text.strip()
    required = {record.pair_id for record in records}
    if set(values) != required:
        missing, extra = len(required - values.keys()), len(values.keys() - required)
        raise ValueError(f"Prediction IDs do not match gold: {missing} missing, {extra} extra")
    return [values[record.pair_id] for record in records]
