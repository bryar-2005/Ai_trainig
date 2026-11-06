"""Validation and CLI failure/success cases for held-out evaluation."""

import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("sacrebleu")
pytest.importorskip("sentencepiece")
pytest.importorskip("regex")
pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from datasets import DatasetDict, load_from_disk

from ml.eval.evaluate import evaluate
from ml.eval.records import load_evaluation_records, load_predictions
from tests.test_eval_pipeline import _write_predictions, make_evaluation_data  # noqa: F401


def test_bad_review_status_duplicate_and_film_are_rejected(evaluation_data):
    dataset, gold, rows = evaluation_data
    rows[0]["review_status"] = ["verified"]
    _write_predictions(gold, rows)
    with pytest.raises(ValueError, match="invalid review_status"):
        load_evaluation_records(dataset, gold, direction="en-ckb")
    rows[0]["review_status"] = "verified"
    _write_predictions(gold, [rows[0], rows[0]])
    with pytest.raises(ValueError, match="Duplicate gold"):
        load_evaluation_records(dataset, gold, direction="en-ckb")
    rows[0]["film_id"] = "different-film"
    _write_predictions(gold, rows)
    with pytest.raises(ValueError, match="film ID"):
        load_evaluation_records(dataset, gold, direction="en-ckb")


def test_split_leakage_and_invalid_timing_are_rejected(evaluation_data, tmp_path):
    dataset, gold, _ = evaluation_data
    original = load_from_disk(str(dataset))
    leaked = DatasetDict({name: split for name, split in original.items()})
    leaked["train"] = leaked["train"].map(lambda row: {"film_id": "test-a"})
    leaked_path = tmp_path / "leaked"
    leaked.save_to_disk(str(leaked_path))
    with pytest.raises(ValueError, match="Film leakage"):
        load_evaluation_records(leaked_path, gold, direction="en-ckb")
    invalid = DatasetDict({name: split for name, split in original.items()})
    invalid["test"] = invalid["test"].map(lambda row: {"english_end_ms": row["english_start_ms"]})
    invalid_path = tmp_path / "invalid"
    invalid.save_to_disk(str(invalid_path))
    with pytest.raises(ValueError, match="positive original source duration"):
        load_evaluation_records(invalid_path, gold, direction="en-ckb")


def test_timing_changes_have_a_separate_fingerprint(evaluation_data, tmp_path):
    dataset, gold, _ = evaluation_data
    _, info = load_evaluation_records(dataset, gold, direction="en-ckb")
    changed = load_from_disk(str(dataset))
    changed["test"] = changed["test"].map(
        lambda row: {"english_end_ms": row["english_end_ms"] + 100}
    )
    changed_path = tmp_path / "changed"
    changed.save_to_disk(str(changed_path))
    _, other = load_evaluation_records(changed_path, gold, direction="en-ckb")
    assert info["dataset_sha256"] == other["dataset_sha256"]
    assert info["test_timing_sha256"] != other["test_timing_sha256"]


def test_prediction_duplicate_and_extra_ids_rejected(evaluation_data, tmp_path):
    dataset, gold, rows = evaluation_data
    records, _ = load_evaluation_records(dataset, gold, direction="en-ckb")
    path = tmp_path / "predictions.jsonl"
    _write_predictions(path, [{"pair_id": rows[0]["pair_id"], "text": "a"}] * 2)
    with pytest.raises(ValueError, match="Duplicate prediction"):
        load_predictions(path, records)
    _write_predictions(
        path,
        [{"pair_id": row["pair_id"], "text": "a"} for row in rows]
        + [{"pair_id": "not-gold", "text": "extra"}],
    )
    with pytest.raises(ValueError, match="1 extra"):
        load_predictions(path, records)


def test_reverse_direction_uses_sorani_as_source(evaluation_data):
    dataset, gold, rows = evaluation_data
    records, _ = load_evaluation_records(dataset, gold, direction="ckb-en")
    assert records[0].source == rows[0]["sorani"]
    assert records[0].reference == rows[0]["english"]
    assert records[0].example.context == ("دەست بە ڕۆیشتن بکە.",)


def test_cli_success_and_usage_error(evaluation_data, tmp_path):
    dataset, gold, rows = evaluation_data
    predictions = tmp_path / "predictions.jsonl"
    _write_predictions(
        predictions, [{"pair_id": row["pair_id"], "text": row["sorani"]} for row in rows]
    )
    output = tmp_path / "successful"
    command = [
        sys.executable,
        "-m",
        "ml.eval.evaluate",
        "--dataset-dir",
        str(dataset),
        "--gold",
        str(gold),
        "--output-dir",
        str(output),
        "--baseline-predictions",
        str(predictions),
        "--candidate-predictions",
        str(predictions),
        "--skip-spbleu",
        "--skip-comet",
        "--label",
        "Offline test report",
    ]
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
    completed = subprocess.run(command, text=True, capture_output=True, env=env, timeout=120)
    assert completed.returncode == 0, completed.stderr
    result = json.loads((output / "metrics.json").read_text())
    assert result["chrfpp_delta"] == 0 and result["label"] == "Offline test report"
    assert "Offline test report" in (output / "comparison.html").read_text(encoding="utf-8")
    assert "glossary adherence was not computed" in completed.stdout
    # Existing outputs cannot be overwritten, and invalid invocation returns 2.
    repeated = subprocess.run(command, text=True, capture_output=True, env=env, timeout=120)
    assert repeated.returncode == 2
    assert "new or empty" in repeated.stderr
    assert (output / "metrics.json").is_file()


def test_output_symlink_and_inputs_inside_output_are_rejected(evaluation_data, tmp_path):
    dataset, gold, rows = evaluation_data
    predictions = tmp_path / "predictions.jsonl"
    _write_predictions(
        predictions, [{"pair_id": row["pair_id"], "text": row["sorani"]} for row in rows]
    )
    link = tmp_path / "report-link"
    link.symlink_to(tmp_path / "not-created", target_is_directory=True)
    args = dict(
        dataset_dir=dataset,
        gold_path=gold,
        direction="en-ckb",
        baseline_predictions=predictions,
        candidate_predictions=predictions,
        skip_spbleu=True,
        skip_comet=True,
    )
    with pytest.raises(ValueError, match="symlink"):
        evaluate(output_dir=link, **args)
    # Empty parents containing input files are never used as an output directory.
    with pytest.raises(ValueError, match="new or empty|overlap"):
        evaluate(output_dir=tmp_path, **args)
