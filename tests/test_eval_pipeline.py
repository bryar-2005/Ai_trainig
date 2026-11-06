import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("sacrebleu")
pytest.importorskip("sentencepiece")
pytest.importorskip("regex")
pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from datasets import Dataset, DatasetDict, disable_progress_bars, load_from_disk

from ml.eval.evaluate import evaluate
from ml.eval.records import load_evaluation_records, load_predictions
from ml.train.config import load_config
from ml.train.train_lora import run_training
from tests.test_eval_metrics import make_sentencepiece_model  # noqa: F401


def _row(film: str, number: int, english: str, sorani: str):
    start = number * 2000
    return {
        "pair_id": f"{film}-{number}",
        "film_id": film,
        "english": english,
        "sorani": sorani,
        "english_start_ms": start,
        "english_end_ms": start + 1500,
        "sorani_start_ms": start,
        "sorani_end_ms": start + 1500,
        "english_speakers": ["JOHN"],
        "sorani_speakers": [],
        "provenance": "human",
        "language_verified": True,
    }


@pytest.fixture(name="evaluation_data")
def make_evaluation_data(tmp_path):
    train = [
        _row("train", 0, "Take the book home.", "کتێبەکە بەرە ماڵەوە."),
        _row("train", 1, "The rain will stop soon.", "بارانەکە بەم زووانە دەوەستێت."),
        _row("train", 2, "We should leave now.", "دەبێت ئێستا بڕۆین."),
    ]
    val = [_row("val", 0, "I will wait here.", "من لێرە چاوەڕێ دەکەم.")]
    test = [
        _row("test-a", 0, "Start walking.", "دەست بە ڕۆیشتن بکە."),
        _row("test-a", 1, "Please keep moving.", "تکایە بەردەوام بە لە ڕۆیشتن."),
        _row("test-a", 2, "Can you see the bridge?", "دەتوانیت پردەکە ببینیت؟"),
        _row("test-b", 0, "The bus is coming.", "پاسەکە دێت."),
    ]
    disable_progress_bars()
    dataset_path = tmp_path / "dataset"
    DatasetDict(
        {
            "train": Dataset.from_list(train),
            "val": Dataset.from_list(val),
            "test": Dataset.from_list(test),
        }
    ).save_to_disk(str(dataset_path))
    gold_path = tmp_path / "gold.jsonl"
    gold_rows = [{**test[1], "review_status": "verified"}, {**test[2], "review_status": "verified"}]
    gold_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in gold_rows), encoding="utf-8"
    )
    return dataset_path, gold_path, gold_rows


def _write_predictions(path, records):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8"
    )


def test_context_uses_full_test_film_not_only_sampled_gold(evaluation_data):
    dataset, gold, _ = evaluation_data
    records, info = load_evaluation_records(dataset, gold, direction="en-ckb")
    assert records[0].example.context == ("[JOHN] Start walking.",)
    assert records[1].example.context[-1] == "[JOHN] Please keep moving."
    assert info["test_pairs"] == 4 and info["scored_pairs"] == 2
    assert all(record.duration_ms == 1500 for record in records)


def test_gold_requires_review_and_original_held_out_source(evaluation_data):
    dataset, gold, rows = evaluation_data
    rows[0]["review_status"] = "unverified"
    _write_predictions(gold, rows)
    with pytest.raises(ValueError, match="unverified"):
        load_evaluation_records(dataset, gold, direction="en-ckb")
    _, info = load_evaluation_records(dataset, gold, direction="en-ckb", allow_unverified_gold=True)
    assert info["unverified_gold_count"] == 1
    rows[0]["review_status"] = "verified"
    rows[0]["english"] = "Altered source."
    _write_predictions(gold, rows)
    with pytest.raises(ValueError, match="source text differs"):
        load_evaluation_records(dataset, gold, direction="en-ckb")


def test_gold_reference_can_be_corrected_and_training_pair_is_rejected(evaluation_data):
    dataset, gold, rows = evaluation_data
    rows[0]["sorani"] = "دەقێکی هەڵسەنگێندراو و چاککراو."
    _write_predictions(gold, rows)
    records, _ = load_evaluation_records(dataset, gold, direction="en-ckb")
    assert records[0].reference == rows[0]["sorani"]
    rows[0]["pair_id"] = "train-0"
    _write_predictions(gold, rows)
    with pytest.raises(ValueError, match="held-out"):
        load_evaluation_records(dataset, gold, direction="en-ckb")


def test_predictions_are_keyed_and_complete(evaluation_data, tmp_path):
    dataset, gold, rows = evaluation_data
    records, _ = load_evaluation_records(dataset, gold, direction="en-ckb")
    prediction = tmp_path / "predictions.jsonl"
    _write_predictions(
        prediction,
        [
            {"pair_id": rows[1]["pair_id"], "text": "two"},
            {"pair_id": rows[0]["pair_id"], "text": "one"},
        ],
    )
    assert load_predictions(prediction, records) == ["one", "two"]
    _write_predictions(prediction, [{"pair_id": rows[0]["pair_id"], "text": "one"}])
    with pytest.raises(ValueError, match="1 missing"):
        load_predictions(prediction, records)


def test_cli_regression_writes_escaped_report_and_exits_one(
    evaluation_data, sentencepiece_model, tmp_path
):
    dataset, gold, rows = evaluation_data
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    _write_predictions(
        baseline, [{"pair_id": row["pair_id"], "text": row["sorani"]} for row in rows]
    )
    _write_predictions(
        candidate,
        [{"pair_id": row["pair_id"], "text": "<script>alert('x')</script>"} for row in rows],
    )
    output = tmp_path / "report"
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
    result = subprocess.run(
        [
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
            str(baseline),
            "--candidate-predictions",
            str(candidate),
            "--spm-model",
            str(sentencepiece_model),
            "--skip-comet",
        ],
        text=True,
        capture_output=True,
        env=env,
        timeout=120,
    )
    assert result.returncode == 1, result.stderr
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["regression"] is True
    assert metrics["baseline"]["chrfpp"] > metrics["candidate"]["chrfpp"]
    assert metrics["baseline"]["spbleu"]["score"] is not None
    page = (output / "comparison.html").read_text(encoding="utf-8")
    assert "&lt;script&gt;" in page and "<script>" not in page
    assert 'dir="rtl"' in page
    assert "explicitly skipped" in page
    assert (output / "predictions.jsonl").is_file()
    records, _ = load_evaluation_records(dataset, gold, direction="en-ckb")
    assert load_predictions(output / "predictions.jsonl", records, field="candidate") == [
        "<script>alert('x')</script>",
        "<script>alert('x')</script>",
    ]


def test_output_safety_and_missing_metrics_fail_closed(evaluation_data, tmp_path):
    dataset, gold, rows = evaluation_data
    preds = tmp_path / "predictions.jsonl"
    _write_predictions(preds, [{"pair_id": row["pair_id"], "text": row["sorani"]} for row in rows])
    args = dict(
        dataset_dir=dataset,
        gold_path=gold,
        direction="en-ckb",
        baseline_predictions=preds,
        candidate_predictions=preds,
        skip_comet=True,
    )
    with pytest.raises(ValueError, match="spm-model"):
        evaluate(output_dir=tmp_path / "report", **args)
    with pytest.raises(ValueError, match="overlap"):
        evaluate(output_dir=dataset / "out", skip_spbleu=True, **args)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep").write_text("important")
    with pytest.raises(ValueError, match="new or empty"):
        evaluate(output_dir=occupied, skip_spbleu=True, **args)
    assert (occupied / "keep").read_text() == "important"


def test_real_tiny_adapter_and_baseline_generate_offline(evaluation_data, tmp_path):
    dataset, gold, _ = evaluation_data
    config = replace(
        load_config(Path("ml/train/config.yaml")),
        dataset_dir=str(dataset),
        output_dir=str(tmp_path / "training"),
    )
    training = run_training(config, smoke_test=True)
    assert training["smoke_test"] is True
    args = dict(
        dataset_dir=dataset,
        gold_path=gold,
        direction="en-ckb",
        skip_spbleu=True,
        skip_comet=True,
        adapter_dir=tmp_path / "training" / "adapter",
        local_files_only=True,
    )
    with pytest.raises(ValueError, match="smoke model"):
        evaluate(output_dir=tmp_path / "refused", **args)
    result = evaluate(output_dir=tmp_path / "model-report", allow_smoke_model=True, **args)
    assert result["model"]["mode"] == "generated"
    assert result["model"]["smoke_model"] is True
    assert result["model"]["examples_with_context"] == 2
    assert len(result["model"]["adapter_config_sha256"]) == 64
    assert len(result["model"]["adapter_assets_sha256"]) == 64
    assert len(result["model"]["training_manifest_sha256"]) == 64
    assert result["report_rows"] == 2
    assert any("random smoke" in warning for warning in result["warnings"])
    assert (tmp_path / "model-report" / "comparison.html").is_file()
    with pytest.raises(ValueError, match="direction does not match"):
        evaluate(
            output_dir=tmp_path / "wrong-direction",
            allow_smoke_model=True,
            **{**args, "direction": "ckb-en"},
        )
    changed = load_from_disk(str(dataset))
    changed["train"] = changed["train"].map(lambda row: {"english": row["english"] + " changed"})
    changed_path = tmp_path / "different-dataset"
    changed.save_to_disk(str(changed_path))
    with pytest.raises(ValueError, match="dataset differs"):
        evaluate(
            output_dir=tmp_path / "wrong-dataset-report",
            allow_smoke_model=True,
            **{**args, "dataset_dir": changed_path},
        )
    wrong_base = tmp_path / "wrong-base"
    wrong_base.mkdir()
    (wrong_base / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        evaluate(
            output_dir=tmp_path / "wrong-base-report",
            base_model_override=str(wrong_base),
            allow_smoke_model=True,
            **args,
        )
