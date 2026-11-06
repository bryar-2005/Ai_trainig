from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from ml.train.config import load_config


def test_default_config_has_required_lora_and_training_values():
    config = load_config(Path("ml/train/config.yaml"))
    assert config.base_model == "facebook/nllb-200-distilled-600M"
    assert config.context_sentences == 3
    assert config.lora.r == 16 and config.lora.alpha == 32 and config.lora.dropout == 0.05
    assert set(config.lora.target_modules) == {
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "fc1",
        "fc2",
    }
    assert config.training.learning_rate == 2e-4
    assert config.training.warmup_steps == 500
    assert config.training.epochs == 3


@pytest.mark.parametrize(
    "contents,expected",
    [
        ("bogus_option: true\n", "Unknown run"),
        ("direction: kurdish-ish\n", "direction"),
        ("direction: [en, ckb]\n", "direction"),
        ("base_revision: ''\n", "base_revision"),
        ("context_sentences: 1\n", "context_sentences"),
        ("lora:\n  r: 0\n", "lora.r"),
        ("training:\n  learning_rate: -2\n", "learning_rate"),
        ("training:\n  mixed_precision: imaginary\n", "mixed_precision"),
        ("training:\n  mixed_precision: [fp32]\n", "mixed_precision"),
        ("training:\n  report_to: [secret_service]\n", "report_to"),
    ],
)
def test_invalid_configs_fail_closed(tmp_path: Path, contents: str, expected: str):
    path = tmp_path / "config.yaml"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match=expected):
        load_config(path)
