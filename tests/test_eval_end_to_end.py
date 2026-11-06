"""Full generated baseline/adapter + real metric APIs, using only random local models."""

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("comet")
pytest.importorskip("sentencepiece")
pytest.importorskip("sacrebleu")
pytest.importorskip("regex")
pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from ml.train.config import load_config
from ml.train.train_lora import run_training
from tests.test_eval_comet import make_tiny_comet_checkpoint  # noqa: F401
from tests.test_eval_metrics import make_sentencepiece_model  # noqa: F401
from tests.test_eval_pipeline import make_evaluation_data  # noqa: F401


def test_adapter_sentencepiece_and_comet_together_offline(
    tiny_comet_checkpoint, sentencepiece_model, evaluation_data, tmp_path
):
    dataset, gold, _ = evaluation_data
    config = replace(
        load_config(Path("ml/train/config.yaml")),
        dataset_dir=str(dataset),
        output_dir=str(tmp_path / "training"),
    )
    run_training(config, smoke_test=True)
    output = tmp_path / "complete-report"
    command = [
        sys.executable,
        "-m",
        "ml.eval.evaluate",
        "--dataset-dir",
        str(dataset),
        "--gold",
        str(gold),
        "--adapter-dir",
        str(tmp_path / "training" / "adapter"),
        "--allow-smoke-model",
        "--local-files-only",
        "--spm-model",
        str(sentencepiece_model),
        "--comet-checkpoint",
        str(tiny_comet_checkpoint),
        "--output-dir",
        str(output),
    ]
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
    }
    completed = subprocess.run(command, text=True, capture_output=True, env=env, timeout=120)
    assert completed.returncode in (0, 1), completed.stderr
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert completed.returncode == (1 if metrics["regression"] else 0)
    assert metrics["model"]["mode"] == "generated" and metrics["model"]["smoke_model"]
    assert metrics["baseline"]["spbleu"] is not None
    assert metrics["candidate"]["comet"]["segment_count"] == 2
    assert (output / "comparison.html").is_file()
    assert (output / "predictions.jsonl").is_file()
