"""Exercise COMET with an original, tiny random local checkpoint, never network assets."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("comet")
pytest.importorskip("sentencepiece")
pytest.importorskip("sacrebleu")
pytest.importorskip("regex")

import pytorch_lightning as lightning
import torch
import yaml
from comet.models.regression.regression_metric import RegressionMetric
from transformers import XLMRobertaConfig, XLMRobertaTokenizerFast

from ml.eval.evaluate import evaluate
from ml.eval.metrics import compute_metrics, load_comet
from tests.test_eval_metrics import make_sentencepiece_model, record  # noqa: F401
from tests.test_eval_pipeline import _write_predictions, make_evaluation_data  # noqa: F401


@pytest.fixture(name="tiny_comet_checkpoint")
def make_tiny_comet_checkpoint(tmp_path: Path, sentencepiece_model: Path) -> Path:
    """Create an actually loadable COMET checkpoint with randomly initialized weights."""
    torch.set_num_threads(min(torch.get_num_threads(), 2))
    encoder = tmp_path / "encoder"
    encoder.mkdir()
    tokenizer = XLMRobertaTokenizerFast(vocab_file=str(sentencepiece_model))
    tokenizer.save_pretrained(encoder)
    XLMRobertaConfig(
        vocab_size=len(tokenizer),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=128,
    ).save_pretrained(encoder)
    metric = RegressionMetric(
        pretrained_model=str(encoder),
        load_pretrained_weights=False,
        local_files_only=True,
        hidden_sizes=[16],
        dropout=0.0,
    )
    root = tmp_path / "comet"
    (root / "checkpoints").mkdir(parents=True)
    (root / "hparams.yaml").write_text(
        yaml.safe_dump({"class_identifier": "regression_metric"}), encoding="utf-8"
    )
    checkpoint = root / "checkpoints" / "model.ckpt"
    torch.save(
        {
            "state_dict": metric.state_dict(),
            "hyper_parameters": dict(metric.hparams),
            "pytorch-lightning_version": lightning.__version__,
        },
        checkpoint,
    )
    return checkpoint


def test_referenceless_comet_qe_is_not_mislabeled_as_reference_based(tmp_path, monkeypatch):
    import comet

    checkpoint = tmp_path / "dummy.ckpt"
    checkpoint.write_bytes(b"test-only loader guard")
    monkeypatch.setattr(
        comet,
        "load_from_checkpoint",
        lambda *args, **kwargs: SimpleNamespace(
            hparams=SimpleNamespace(class_identifier="referenceless_regression_metric")
        ),
    )
    with pytest.raises(ValueError, match="reference-based"):
        load_comet(checkpoint, local_files_only=True)


def test_real_comet_local_checkpoint_predicts_offline(tiny_comet_checkpoint):
    model = load_comet(tiny_comet_checkpoint, local_files_only=True)
    records = [record("a", "Hello my friend", "سڵاو هاوڕێکم")]
    scores = compute_metrics(
        records, ["سڵاو هاوڕێکم"], direction="en-ckb", comet_model=model, device="cpu"
    )["comet"]
    assert scores["segment_count"] == 1
    assert scores["requires_references"] is True
    assert scores["model_class"] == "RegressionMetric"
    assert scores["package_version"]
    assert math.isfinite(scores["score"])


def test_full_evaluator_uses_real_comet_and_sentencepiece(
    tiny_comet_checkpoint, sentencepiece_model, evaluation_data, tmp_path
):
    dataset, gold, rows = evaluation_data
    predictions = tmp_path / "input.jsonl"
    _write_predictions(
        predictions, [{"pair_id": row["pair_id"], "text": row["sorani"]} for row in rows]
    )
    result = evaluate(
        dataset_dir=dataset,
        gold_path=gold,
        output_dir=tmp_path / "report",
        direction="en-ckb",
        baseline_predictions=predictions,
        candidate_predictions=predictions,
        sentencepiece_model=sentencepiece_model,
        comet_checkpoint=tiny_comet_checkpoint,
        local_files_only=True,
    )
    assert result["regression"] is False
    assert result["baseline"]["spbleu"]["score"] > 0
    assert math.isfinite(result["baseline"]["comet"]["score"])
    assert result["baseline"]["comet"] == result["candidate"]["comet"]
    stored = json.loads((tmp_path / "report" / "metrics.json").read_text(encoding="utf-8"))
    assert stored["comet_checkpoint_sha256"] and stored["comet_hparams_sha256"]
    assert stored["candidate"]["comet"]["segment_count"] == 2
    assert any("Sorani" in warning for warning in stored["warnings"])

    # Exercise the public CLI with both real metric implementations as well.
    command = [
        sys.executable,
        "-m",
        "ml.eval.evaluate",
        "--dataset-dir",
        str(dataset),
        "--gold",
        str(gold),
        "--output-dir",
        str(tmp_path / "cli-report"),
        "--baseline-predictions",
        str(predictions),
        "--candidate-predictions",
        str(predictions),
        "--spm-model",
        str(sentencepiece_model),
        "--comet-checkpoint",
        str(tiny_comet_checkpoint),
        "--local-files-only",
    ]
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "OMP_NUM_THREADS": "2"}
    cli = subprocess.run(command, text=True, capture_output=True, env=env, timeout=120)
    assert cli.returncode == 0, cli.stderr
    cli_metrics = json.loads((tmp_path / "cli-report" / "metrics.json").read_text())
    assert cli_metrics["baseline"]["comet"]["segment_count"] == 2
    assert cli_metrics["baseline"]["spbleu"]["score"] > 0
