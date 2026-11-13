"""Offline overlap audit integration, using synthetic fixtures only."""

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict, disable_progress_bars, load_from_disk

import ml.data.audit_overlap as audit
from ml.data.audit_overlap import OverlapAuditConfig, OverlapAuditError, audit_overlap
from ml.data.overlap_matching import OverlapLimitError, WorkBudget
from ml.data.stage_text_pairs import TextPairStagingConfig, stage_text_pairs
from ml.data.stage_tmx_pairs import TmxPairStagingConfig, _demo_files, stage_tmx_pairs


def row(film: str, number: int, english: str, sorani: str, **extra) -> dict:
    return {
        "pair_id": f"synthetic-{film}-{number}",
        "film_id": f"synthetic-{film}",
        "english": english,
        "sorani": sorani,
        "english_start_ms": number * 1000,
        "sorani_start_ms": number * 1000,
        "provenance": "unknown",
        "language_verified": False,
        **extra,
    }


def mock_splits() -> dict[str, list[dict]]:
    return {
        "train": [row("train", 1, "The tiny kettle whistles gently.", "دەقی تاقیکردنەوەی یەکەم.")],
        "val": [row("val", 1, "Our purple compass points west.", "جۆرێکی جیاواز بۆ دوو.")],
        "test": [row("test", 1, "A silver moon rises over hills.", "هەوا بۆ تاقیکردنەوەی سێ.")],
    }


def dataset(directory: Path, splits: dict[str, list[dict]] | None = None) -> Path:
    disable_progress_bars()
    splits = splits or mock_splits()
    result = directory / "dataset"
    result.parent.mkdir(parents=True, exist_ok=True)
    nonempty_rows = next(rows for rows in splits.values() if rows)
    empty_features = Dataset.from_list(nonempty_rows).features
    built = {
        name: (
            Dataset.from_list(rows)
            if rows
            else Dataset.from_dict(
                {column: [] for column in empty_features}, features=empty_features
            )
        )
        for name, rows in splits.items()
    }
    empty_shards = {name: 1 for name, rows in splits.items() if not rows}
    DatasetDict(built).save_to_disk(str(result), num_shards=empty_shards)
    return result


def manifest(source_id: str = "synthetic-a") -> dict:
    return {
        "source_id": source_id,
        "source_revision": "fixture-1",
        "source_reference": "Independently authored synthetic fixture",
        "provenance": "unknown",
        "rights": {
            "status": "synthetic",
            "affirmed_by_user": True,
            "original_text_evidence": "Synthetic testing only",
            "translation_evidence": "Synthetic testing only",
            "license_or_permission": "Synthetic testing only",
            "attribution": "No external material",
            "permitted_uses": ["synthetic_testing"],
        },
    }


def stage(
    directory: Path,
    english: str = "Synthetic stage example one.\nAnother synthetic example.\n",
    sorani: str = "دەقی تاقیکردنەوەی چوار.\nدەقی تاقیکردنەوەی پێنج.\n",
) -> TextPairStagingConfig:
    raw = directory / "raw"
    raw.mkdir(parents=True)
    en, ckb, source = raw / "en.txt", raw / "ckb.txt", raw / "source.json"
    en.write_text(english, encoding="utf-8")
    ckb.write_text(sorani, encoding="utf-8")
    source.write_text(json.dumps(manifest(), ensure_ascii=False), encoding="utf-8")
    config = TextPairStagingConfig(en, ckb, source, directory / "stage")
    stage_text_pairs(config)
    return config


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def cli(*arguments: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
    return subprocess.run(
        [sys.executable, "-m", "ml.data.audit_overlap", *arguments],
        text=True,
        capture_output=True,
        env=env,
        timeout=60,
        check=False,
    )


def test_demo_finds_expected_signals_without_publishing_dialogue(tmp_path: Path):
    output = tmp_path / "demo"
    completed = cli("--demo", "--output-dir", str(output))
    # The demo intentionally contains overlaps; exit 1 requests review, not a crash.
    assert completed.returncode == 1, completed.stderr
    summary = json.loads(completed.stdout)
    report = json.loads((output / "overlap-report.json").read_text(encoding="utf-8"))
    details = read_jsonl(output / "matches.jsonl")
    assert summary["status"] == report["status"] == "review_needed"
    assert report["schema"] == "sorani-pair-overlap-audit-v1"
    assert report["training_ready"] is False and summary["training_ready"] is False
    assert report["comparisons"]["train_vs_test"]["signals"]["bilingual_exact"] == 1
    staged = report["comparisons"]["staging-1_vs_test"]
    assert staged["signals"]["bilingual_exact"] == staged["signals"]["bilingual_near"] == 1
    assert report["staging_inputs"][0]["declared_rights_status"] == "synthetic"
    assert report["gold_not_supplied"] == ["en-ckb", "ckb-en"]
    assert len(details) == report["details"]["total_records"] == 3
    assert (
        report["details"]["sha256"]
        == hashlib.sha256((output / "matches.jsonl").read_bytes()).hexdigest()
    )
    assert "Please open" not in completed.stdout
    assert "PLEASE  OPEN" not in (output / "overlap-report.json").read_text(encoding="utf-8")
    assert "PLEASE  OPEN" not in (output / "matches.jsonl").read_text(encoding="utf-8")


def test_no_signals_is_not_training_approval_and_report_is_deterministic(tmp_path: Path):
    path = dataset(tmp_path)
    first = audit_overlap(OverlapAuditConfig(path, tmp_path / "first"))
    second = audit_overlap(OverlapAuditConfig(path, tmp_path / "second"))
    assert first == second
    assert first["status"] == "no_signals_detected"
    assert first["training_ready"] is False
    assert first["gold_not_supplied"] == ["en-ckb", "ckb-en"]
    assert all(c["matched_row_pairs"] == 0 for c in first["comparisons"].values())
    assert first["details"]["total_records"] == 0
    completed = cli("--dataset-dir", str(path), "--output-dir", str(tmp_path / "cli"))
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["status"] == "no_signals_detected"
    assert "not establish independence or rights" in completed.stdout


def test_staged_only_candidate_allows_empty_timed_train_without_fabricating_rows(tmp_path: Path):
    splits = mock_splits()
    splits["train"] = []
    heldout = splits["test"][0]
    staged = stage(tmp_path / "text", heldout["english"] + "\n", heldout["sorani"] + "\n")
    path = dataset(tmp_path, splits)
    with pytest.raises(OverlapAuditError, match="empty train split requires"):
        audit_overlap(OverlapAuditConfig(path, tmp_path / "without-stage"))
    report = audit_overlap(OverlapAuditConfig(path, tmp_path / "out", (staged.output_dir,)))
    assert report["dataset"]["empty_train_split"] is True
    assert report["dataset"]["split_rows"] == {"train": 0, "val": 1, "test": 1}
    assert report["comparisons"]["train_vs_val"]["left_rows"] == 0
    assert report["comparisons"]["staging-1_vs_test"]["signals"]["bilingual_exact"] == 1
    assert report["training_ready"] is False and len(load_from_disk(str(path))["train"]) == 0


def test_empty_train_mode_still_reports_shared_heldout_films_and_rejects_duplicate_ids(
    tmp_path: Path,
):
    splits = mock_splits()
    splits["train"] = []
    splits["val"][0]["film_id"] = splits["test"][0]["film_id"]
    staged = stage(tmp_path / "text")
    path = dataset(tmp_path / "same-film", splits)
    report = audit_overlap(OverlapAuditConfig(path, tmp_path / "out", (staged.output_dir,)))
    assert report["comparisons"]["val_vs_test"]["shared_film_ids"] == 1
    assert report["status"] == "review_needed"
    splits["val"][0]["pair_id"] = splits["test"][0]["pair_id"]
    duplicate = dataset(tmp_path / "duplicate-id", splits)
    with pytest.raises(OverlapAuditError, match="split or row validation"):
        audit_overlap(OverlapAuditConfig(duplicate, tmp_path / "bad", (staged.output_dir,)))
    assert not (tmp_path / "bad").exists()


def test_shared_film_ids_are_reported_for_all_splits_without_text_overlap(tmp_path: Path):
    splits = mock_splits()
    for rows in splits.values():
        rows[0]["film_id"] = "synthetic-shared"
    result = audit_overlap(OverlapAuditConfig(dataset(tmp_path, splits), tmp_path / "out"))
    assert result["status"] == "review_needed"
    assert all(item["shared_film_ids"] == 1 for item in result["comparisons"].values())
    details = read_jsonl(tmp_path / "out" / "matches.jsonl")
    assert len(details) == 3 and all(item["type"] == "shared_film_id" for item in details)


def test_both_review_stagers_are_accepted_and_quarantine_is_not_scanned(tmp_path: Path):
    text = stage(tmp_path / "text", "One fictional message.\n\n", "دەقی خەیاڵی.\nدووەم.\n")
    tmx_root = tmp_path / "tmx" / "raw"
    tmx_root.mkdir(parents=True)
    tmx, source = _demo_files(tmx_root)
    tmx_out = tmp_path / "tmx" / "out"
    stage_tmx_pairs(TmxPairStagingConfig(tmx, source, tmx_out))
    output = tmp_path / "audit"
    report = audit_overlap(
        OverlapAuditConfig(dataset(tmp_path), output, (text.output_dir, tmx_out))
    )
    assert report["staging_inputs"][0]["staged_pairs"] == 1
    assert report["staging_inputs"][0]["quarantined_not_compared"] == 1
    assert report["staging_inputs"][1]["schema"] == "sorani-tmx-pair-staging-v1"
    assert report["comparisons"]["staging-1_vs_val"]["left_rows"] == 1
    assert report["comparisons"]["staging-2_vs_test"]["left_rows"] == 3
    assert report["training_ready"] is False
    assert text.english_file.read_text(encoding="utf-8") == "One fictional message.\n\n"


def test_reused_pair_id_is_reported_even_if_the_text_has_changed(tmp_path: Path):
    staged = stage(tmp_path / "text")
    stage_id = read_jsonl(staged.output_dir / "pairs.jsonl")[0]["pair_id"]
    splits = mock_splits()
    splits["test"][0]["pair_id"] = stage_id
    report = audit_overlap(
        OverlapAuditConfig(dataset(tmp_path, splits), tmp_path / "out", (staged.output_dir,))
    )
    assert report["comparisons"]["staging-1_vs_test"]["signals"]["same_pair_id"] == 1
    assert report["status"] == "review_needed"


def write_gold(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows), encoding="utf-8"
    )


def test_both_corrected_gold_directions_are_compared_as_separate_variants(tmp_path: Path):
    splits = mock_splits()
    test = splits["test"][0]
    splits["train"][0]["english"] = "Corrected English reference from a synthetic review."
    splits["train"][0]["sorani"] = "دەقی چاککراوی تاقیکردنەوە."
    forward = {**test, "sorani": splits["train"][0]["sorani"], "review_status": "verified"}
    reverse = {**test, "english": splits["train"][0]["english"], "review_status": "unverified"}
    en_gold, ckb_gold = tmp_path / "en.jsonl", tmp_path / "ckb.jsonl"
    write_gold(en_gold, [forward])
    write_gold(ckb_gold, [reverse])
    path = dataset(tmp_path, splits)
    report = audit_overlap(OverlapAuditConfig(path, tmp_path / "out", (), en_gold, ckb_gold))
    assert report["comparisons"]["train_vs_test"]["matched_row_pairs"] == 0
    assert report["comparisons"]["train_vs_test-gold-en-ckb"]["signals"]["sorani_exact"] == 1
    assert report["comparisons"]["train_vs_test-gold-ckb-en"]["signals"]["english_exact"] == 1
    assert report["gold_not_supplied"] == []
    assert [gold["unverified_rows"] for gold in report["gold_inputs"]] == [0, 1]
    assert report["gold_inputs"][0]["sha256"] == hashlib.sha256(en_gold.read_bytes()).hexdigest()


@pytest.mark.parametrize("problem", ["unknown", "wrong-film", "duplicate", "source", "status"])
def test_invalid_gold_is_rejected_without_output(tmp_path: Path, problem: str):
    splits = mock_splits()
    test = {**splits["test"][0], "review_status": "verified"}
    values = [test]
    if problem == "unknown":
        test["pair_id"] = "missing"
    elif problem == "wrong-film":
        test["film_id"] = "another-film"
    elif problem == "duplicate":
        values.append(dict(test))
    elif problem == "source":
        test["english"] = "Changed synthetic source"
    else:
        test["review_status"] = ["verified"]
    gold = tmp_path / "gold.jsonl"
    write_gold(gold, values)
    output = tmp_path / "out"
    with pytest.raises(OverlapAuditError):
        audit_overlap(OverlapAuditConfig(dataset(tmp_path, splits), output, (), gold))
    assert not output.exists()
    assert not list(tmp_path.glob(".out.audit-*"))


def test_gold_json_is_strict_about_duplicate_keys_and_nonfinite_numbers(tmp_path: Path):
    path = dataset(tmp_path)
    gold = tmp_path / "gold.jsonl"
    for i, raw in enumerate((b'{"pair_id":"a","pair_id":"b"}\n', b'{"x":NaN}\n', b'{"x":1e999}\n')):
        gold.write_bytes(raw)
        with pytest.raises(OverlapAuditError, match="Invalid UTF-8 JSON"):
            audit_overlap(OverlapAuditConfig(path, tmp_path / f"out-{i}", (), gold))
        assert not (tmp_path / f"out-{i}").exists()


def test_explicit_context_and_extra_dataset_splits_fail_closed(tmp_path: Path):
    splits = mock_splits()
    splits["train"][0]["source_context"] = ["Synthetic context outside the pair."]
    with pytest.raises(OverlapAuditError, match="Explicit source_context"):
        audit_overlap(OverlapAuditConfig(dataset(tmp_path / "context", splits), tmp_path / "out"))
    extra = mock_splits()
    extra["extra"] = [row("extra", 1, "Another sentence.", "دەقێکی تر.")]
    with pytest.raises(OverlapAuditError, match="exactly train, val and test"):
        audit_overlap(OverlapAuditConfig(dataset(tmp_path / "extra", extra), tmp_path / "out2"))
    assert not (tmp_path / "out").exists() and not (tmp_path / "out2").exists()


def test_incomplete_or_corrupted_stage_is_rejected_before_publishing(tmp_path: Path):
    config = stage(tmp_path / "text")
    path = dataset(tmp_path)
    pairs = config.output_dir / "pairs.jsonl"
    pairs.write_bytes(pairs.read_bytes() + b"\n")
    output = tmp_path / "out"
    with pytest.raises(OverlapAuditError, match="hashes"):
        audit_overlap(OverlapAuditConfig(path, output, (config.output_dir,)))
    assert not output.exists()


@pytest.mark.parametrize("invalid_schema", [[], None, "other-format"])
def test_malformed_staging_schema_is_a_controlled_error(tmp_path: Path, invalid_schema):
    staged = stage(tmp_path / "text")
    report_path = staged.output_dir / "staging-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["schema"] = invalid_schema
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(OverlapAuditError, match="unsupported or incomplete"):
        audit_overlap(OverlapAuditConfig(dataset(tmp_path), tmp_path / "out", (staged.output_dir,)))
    assert not (tmp_path / "out").exists()


def test_staging_inputs_cannot_be_hardlinks_to_each_other(tmp_path: Path):
    staged = stage(tmp_path / "text")
    quarantine = staged.output_dir / "quarantine.jsonl"
    quarantine.unlink()
    try:
        os.link(staged.output_dir / "pairs.jsonl", quarantine)
    except OSError:
        pytest.skip("Creating hardlinks is not permitted on this platform")
    with pytest.raises(OverlapAuditError, match="must be distinct"):
        audit_overlap(OverlapAuditConfig(dataset(tmp_path), tmp_path / "out", (staged.output_dir,)))
    assert not (tmp_path / "out").exists()


def test_report_source_boolean_cannot_be_replaced_with_an_integer(tmp_path: Path):
    staged = stage(tmp_path / "text")
    report_path = staged.output_dir / "staging-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["source"]["rights"]["affirmed_by_user"] = 1
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(OverlapAuditError, match="source declaration"):
        audit_overlap(OverlapAuditConfig(dataset(tmp_path), tmp_path / "out", (staged.output_dir,)))
    assert not (tmp_path / "out").exists()


def test_no_valid_staged_pairs_does_not_produce_a_clean_report(tmp_path: Path):
    staged = stage(tmp_path / "text", "\n", "دەق.\n")
    with pytest.raises(OverlapAuditError, match="counts are empty"):
        audit_overlap(OverlapAuditConfig(dataset(tmp_path), tmp_path / "out", (staged.output_dir,)))
    assert not (tmp_path / "out").exists()


def test_dataset_byte_limit_is_checked_before_expanding_rows(tmp_path: Path, monkeypatch):
    path = dataset(tmp_path)
    monkeypatch.setattr(audit, "_MAX_DATASET_BYTES", 1)
    with pytest.raises(OverlapAuditError, match="in-memory byte limit"):
        audit_overlap(OverlapAuditConfig(path, tmp_path / "out"))
    assert not (tmp_path / "out").exists()


def test_pair_identity_is_checked_even_if_the_staging_hash_is_rewritten(tmp_path: Path):
    staged = stage(tmp_path / "text")
    pairs_path = staged.output_dir / "pairs.jsonl"
    pairs = read_jsonl(pairs_path)
    pairs[0]["pair_id"] = "0" * 64
    write_gold(pairs_path, pairs)
    report_path = staged.output_dir / "staging-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["outputs"]["pairs_sha256"] = hashlib.sha256(pairs_path.read_bytes()).hexdigest()
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(OverlapAuditError, match="source identity"):
        audit_overlap(OverlapAuditConfig(dataset(tmp_path), tmp_path / "out", (staged.output_dir,)))
    assert not (tmp_path / "out").exists()


def test_duplicate_staged_ids_across_distinct_outputs_abort(tmp_path: Path):
    first = stage(tmp_path / "text")
    second = replace(first, output_dir=tmp_path / "second-stage")
    stage_text_pairs(second)
    output = tmp_path / "out"
    with pytest.raises(OverlapAuditError, match="Duplicate staged pair ID"):
        audit_overlap(
            OverlapAuditConfig(dataset(tmp_path), output, (first.output_dir, second.output_dir))
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".out.audit-*"))


def test_staging_changed_after_read_is_detected(tmp_path: Path, monkeypatch):
    staged = stage(tmp_path / "text")
    original = audit._stage_rows

    def mutate_after_read(stage_input, seen):
        yield from original(stage_input, seen)
        (stage_input.directory / "quarantine.jsonl").write_bytes(b"changed\n")

    monkeypatch.setattr(audit, "_stage_rows", mutate_after_read)
    output = tmp_path / "out"
    with pytest.raises(OverlapAuditError, match="staging input changed"):
        audit_overlap(OverlapAuditConfig(dataset(tmp_path), output, (staged.output_dir,)))
    assert not output.exists()
    assert not list(tmp_path.glob(".out.audit-*"))


def test_changed_dataset_and_work_budget_abort_without_publishing(tmp_path: Path, monkeypatch):
    path = dataset(tmp_path)
    original = audit._dataset
    calls = 0

    def changed_dataset(dataset_path, *, allow_empty_train=False):
        nonlocal calls
        calls += 1
        splits, fingerprint = original(dataset_path, allow_empty_train=allow_empty_train)
        return splits, fingerprint if calls == 1 else "0" * 64

    monkeypatch.setattr(audit, "_dataset", changed_dataset)
    with pytest.raises(OverlapAuditError, match="Dataset changed"):
        audit_overlap(OverlapAuditConfig(path, tmp_path / "changed"))
    assert not (tmp_path / "changed").exists()
    monkeypatch.undo()
    monkeypatch.setattr(audit, "WorkBudget", lambda: WorkBudget(max_reference_windows=0))
    with pytest.raises(OverlapLimitError, match="reference windows"):
        audit_overlap(OverlapAuditConfig(path, tmp_path / "limited"))
    assert not (tmp_path / "limited").exists()


def test_detail_cap_is_explicit_and_does_not_change_aggregate_counts(tmp_path: Path, monkeypatch):
    root = tmp_path / "mock"
    root.mkdir()
    en, ckb, source = audit._demo_inputs(root)
    staging = root / "stage"
    stage_text_pairs(TextPairStagingConfig(en, ckb, source, staging))
    path = audit._demo_dataset(root)
    monkeypatch.setattr(audit, "_MAX_DETAIL_RECORDS", 1)
    report = audit_overlap(OverlapAuditConfig(path, tmp_path / "out", (staging,)))
    assert report["status"] == "review_needed"
    assert report["details"]["total_records"] == 3
    assert report["details"]["written_records"] == 1 and report["details"]["truncated"] is True
    assert report["comparisons"]["staging-1_vs_test"]["matched_row_pairs"] == 2
    assert len(read_jsonl(tmp_path / "out" / "matches.jsonl")) == 1


def test_no_overwrite_path_overlap_and_racing_output_are_safe(tmp_path: Path, monkeypatch):
    path = dataset(tmp_path)
    staged = stage(tmp_path / "text")
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "keep.txt").write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(OverlapAuditError, match="already exists"):
        audit_overlap(OverlapAuditConfig(path, existing))
    with pytest.raises(OverlapAuditError, match="must not contain"):
        audit_overlap(OverlapAuditConfig(path, path / "child"))
    with pytest.raises(OverlapAuditError, match="same staging"):
        audit_overlap(OverlapAuditConfig(path, tmp_path / "bad", (staged.output_dir,) * 2))
    try:
        link = tmp_path / "out-link"
        link.symlink_to(tmp_path / "missing", target_is_directory=True)
    except OSError:
        pytest.skip("Creating symlinks is not permitted on this platform")
    with pytest.raises(OverlapAuditError, match="already exists"):
        audit_overlap(OverlapAuditConfig(path, link))
    real_publish = audit._publish

    def competing_process(temp, output):
        output.mkdir()
        (output / "keep.txt").write_text("someone else's file", encoding="utf-8")
        real_publish(temp, output)

    monkeypatch.setattr(audit, "_publish", competing_process)
    race = tmp_path / "race"
    with pytest.raises(OverlapAuditError, match="already exists"):
        audit_overlap(OverlapAuditConfig(path, race))
    assert (race / "keep.txt").read_text(encoding="utf-8") == "someone else's file"
    assert (existing / "keep.txt").read_text(encoding="utf-8") == "do not overwrite"
    assert not list(tmp_path.glob(".race.audit-*"))


def test_cli_rejects_invalid_input_combinations_without_output(tmp_path: Path):
    output = tmp_path / "out"
    missing = cli("--output-dir", str(output))
    assert missing.returncode == 2 and "--dataset-dir" in missing.stderr
    combo = cli("--demo", "--dataset-dir", "unused", "--output-dir", str(output))
    assert combo.returncode == 2 and "cannot be combined" in combo.stderr
    assert not output.exists()
