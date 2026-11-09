import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("sacrebleu")

import torch
import yaml
from datasets import Dataset, DatasetDict, disable_progress_bars
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from ml.data.build_dataset import BuildConfig, build_dataset
from ml.train import train_lora
from ml.train.config import load_config
from ml.train.dataset import make_context_examples
from ml.train.merge_adapter import merge_adapter
from ml.train.model_utils import create_smoke_base, resize_embeddings_for_tokenizer
from ml.train.train_lora import run_training


def _rows(split: str, count: int) -> Dataset:
    records = [
        {
            "pair_id": f"{split}-{i}",
            "film_id": f"{split}-{i // 50}",
            "english": f"Please find object number {i} and bring it back.",
            "sorani": f"تکایە ئەو شتە بدۆزەوە و بیگەڕێنەوە {i}.",
            "english_start_ms": i * 2000,
            "sorani_start_ms": i * 2000,
            "english_speakers": ["JOHN"],
            "sorani_speakers": [],
            "provenance": ["human", "unknown", "machine"][i % 3],
            # The languages of these authored synthetic fixtures are known.
            "language_verified": True,
        }
        for i in range(count)
    ]
    return Dataset.from_list(records)


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("lora-smoke")
    disable_progress_bars()
    dataset = DatasetDict(
        {name: _rows(name, count) for name, count in (("train", 200), ("val", 20), ("test", 20))}
    )
    dataset.save_to_disk(str(root / "dataset"))
    config = replace(
        load_config(Path("ml/train/config.yaml")),
        dataset_dir=str(root / "dataset"),
        output_dir=str(root / "run"),
    )
    started = time.monotonic()
    manifest = run_training(config, smoke_test=True, limit=200)
    assert time.monotonic() - started < 300
    return root, config, manifest


def test_smoke_trains_evaluates_and_saves_full_checkpoint(smoke_run):
    root, _, manifest = smoke_run
    assert manifest["status"] == "completed"
    assert manifest["direction"] == "en-ckb"
    assert manifest["smoke_test"] is True
    assert manifest["train"]["samples"] == 200
    assert manifest["train"]["with_previous_context"] > 0
    assert manifest["global_step"] == 13
    assert len(manifest["base_model_sha256"]) == 64
    assert manifest["base_revision_resolved"] is None
    assert isinstance(manifest["best_chrf"], float)
    assert any("random" in warning for warning in manifest["warnings"])
    checkpoint = root / "run" / "checkpoint-5"
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"):
        assert (checkpoint / name).is_file()
    weights = load_file(str(root / "run" / "adapter" / "adapter_model.safetensors"))
    assert any("trainable_tokens_delta" in key for key in weights)
    assert not any(key.endswith("shared.weight") for key in weights)
    assert (root / "run" / "adapter" / "training_manifest.json").is_file()


def test_resuming_restores_model_optimizer_scheduler_rng_and_callback_state(smoke_run):
    root, config, _ = smoke_run
    adapter = root / "run" / "adapter" / "adapter_model.safetensors"
    before = load_file(str(adapter))
    manifest = run_training(
        config, smoke_test=True, limit=200, resume=str(root / "run" / "checkpoint-5")
    )
    after = load_file(str(adapter))
    assert manifest["global_step"] == 13
    assert before.keys() == after.keys()
    assert all(torch.equal(before[key], after[key]) for key in before)


def test_continue_adapter_trains_existing_weights_and_reports_progress(smoke_run, tmp_path):
    root, config, manifest = smoke_run
    initial = root / "run" / "adapter"
    continued = replace(
        train_lora._smoke_config(config),
        base_model=str(root / "run" / "smoke_base"),
        output_dir=str(tmp_path / "continued"),
        local_files_only=True,
    )
    events = []
    result = run_training(continued, initial_adapter=initial, progress=events.append)
    assert result["status"] == "completed" and result["smoke_test"] is True
    assert result["initial_training_identity_sha256"] == manifest["identity_sha256"]
    assert any(event["stage"] == "training" for event in events)
    before = load_file(str(initial / "adapter_model.safetensors"))
    after = load_file(str(tmp_path / "continued" / "adapter" / "adapter_model.safetensors"))
    assert before.keys() == after.keys()
    assert any(not torch.equal(before[key], after[key]) for key in before)
    wrong = replace(continued, direction="ckb-en", output_dir=str(tmp_path / "wrong"))
    with pytest.raises(ValueError, match="direction"):
        run_training(wrong, initial_adapter=initial)


def test_continued_cached_hub_adapter_never_attempts_network(smoke_run, tmp_path, monkeypatch):
    import huggingface_hub.constants
    import requests
    import transformers.utils.hub

    root, config, _ = smoke_run
    commit = "c" * 40
    cache = tmp_path / "cache"
    snapshot = cache / "models--local--base" / "snapshots" / commit
    snapshot.parent.mkdir(parents=True)
    shutil.copytree(root / "run" / "smoke_base", snapshot)
    adapter = tmp_path / "adapter"
    shutil.copytree(root / "run" / "adapter", adapter)
    manifest_path = adapter / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        actual_base_model="local/base", base_model_sha256=None, base_revision_resolved=commit
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    continued = replace(
        train_lora._smoke_config(config),
        base_model="local/base",
        base_revision=commit,
        output_dir=str(tmp_path / "continued-hub"),
        local_files_only=True,
    )
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(cache))
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(transformers.utils.hub, "_is_offline_mode", False)

    def deny_http(*args, **kwargs):
        raise AssertionError("offline continued training attempted a network request")

    monkeypatch.setattr(requests.sessions.Session, "request", deny_http)
    result = run_training(continued, initial_adapter=adapter)
    assert result["status"] == "completed" and result["base_revision_resolved"] == commit


def test_alias_checkpoint_works_for_continued_training_merge_and_evaluation(smoke_run, tmp_path):
    from safetensors.torch import save_file

    from ml.eval.model_compare import generate_comparison
    from ml.eval.records import EvaluationRecord
    from ml.train.model_utils import local_model_digest

    root, config, training_manifest = smoke_run
    original_base = root / "run" / "smoke_base"
    original_adapter = root / "run" / "adapter"
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    shutil.copytree(original_base, base)
    shutil.copytree(original_adapter, adapter)

    weights_path = base / "model.safetensors"
    weights = load_file(str(weights_path))
    retained = "model.decoder.embed_tokens.weight"
    weights[retained] = weights.pop("model.shared.weight").clone()
    replacement = base / "replacement.safetensors"
    save_file(
        weights,
        str(replacement),
        metadata={
            "format": "pt",
            "model.shared.weight": retained,
            "model.encoder.embed_tokens.weight": retained,
            "lm_head.weight": retained,
        },
    )
    replacement.replace(weights_path)
    saved_checkpoint = weights_path.read_bytes()
    manifest_path = adapter / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(actual_base_model=str(base), base_model_sha256=local_model_digest(base))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    row = _rows("test", 1)[0]
    example = make_context_examples([row], direction="en-ckb")[0]
    records = [
        EvaluationRecord(
            pair_id=row["pair_id"],
            film_id=row["film_id"],
            source=row["english"],
            reference=row["sorani"],
            duration_ms=2000,
            provenance=row["provenance"],
            example=example,
        )
    ]
    arguments = {
        "direction": "en-ckb",
        "dataset_sha256": training_manifest["dataset_sha256"],
        "local_files_only": True,
        "allow_smoke_model": True,
    }
    expected_base, expected_adapter, _ = generate_comparison(
        records, adapter_dir=original_adapter, **arguments
    )
    actual_base, actual_adapter, info = generate_comparison(
        records, adapter_dir=adapter, **arguments
    )
    assert (actual_base, actual_adapter) == (expected_base, expected_adapter)
    assert info["smoke_model"]

    merged = tmp_path / "merged"
    report = merge_adapter(adapter, merged, local_files_only=True)
    assert report["base_identity_verified"] and report["smoke_test"]
    merged_model = AutoModelForSeq2SeqLM.from_pretrained(str(merged), local_files_only=True).eval()
    reference_base = AutoModelForSeq2SeqLM.from_pretrained(
        str(original_base), local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(str(original_adapter), local_files_only=True)
    resize_embeddings_for_tokenizer(reference_base, tokenizer)
    reference = PeftModel.from_pretrained(reference_base, str(original_adapter)).eval()
    tokens = torch.tensor([[training_manifest["source_language_id"], tokenizer.eos_token_id]])
    decoder = torch.tensor([[tokenizer.eos_token_id, training_manifest["target_language_id"]]])
    with torch.inference_mode():
        expected = reference(input_ids=tokens, decoder_input_ids=decoder).logits
        actual = merged_model(input_ids=tokens, decoder_input_ids=decoder).logits
    assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-5)

    continued = replace(
        train_lora._smoke_config(config),
        base_model=str(base),
        output_dir=str(tmp_path / "continued"),
        local_files_only=True,
    )
    result = run_training(continued, initial_adapter=adapter)
    assert result["status"] == "completed" and result["smoke_test"]
    assert weights_path.read_bytes() == saved_checkpoint


def test_resume_rejects_changed_config_and_output_reuse(smoke_run):
    root, config, _ = smoke_run
    with pytest.raises(ValueError, match="differs"):
        run_training(
            replace(config, direction="ckb-en"),
            smoke_test=True,
            resume=str(root / "run" / "checkpoint-5"),
        )
    with pytest.raises(ValueError, match="not empty"):
        run_training(config, smoke_test=True)
    assert (
        json.loads((root / "run" / "training_manifest.json").read_text())["status"] == "completed"
    )


def test_merged_model_matches_adapter_logits_and_language_bos(smoke_run):
    root, _, manifest = smoke_run
    adapter_path = root / "run" / "adapter"
    merged_path = root / "merged"
    report = merge_adapter(adapter_path, merged_path, local_files_only=True)
    assert report["smoke_test"] is True
    assert report["base_identity_verified"] is True
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), local_files_only=True)
    base = AutoModelForSeq2SeqLM.from_pretrained(
        str(root / "run" / "smoke_base"), local_files_only=True
    )
    resize_embeddings_for_tokenizer(base, tokenizer)
    adapted = PeftModel.from_pretrained(base, str(adapter_path)).eval()
    merged = AutoModelForSeq2SeqLM.from_pretrained(str(merged_path), local_files_only=True).eval()
    source = torch.tensor(
        [[manifest["source_language_id"], manifest["context_token_id"], tokenizer.eos_token_id]]
    )
    decoder = torch.tensor(
        [[tokenizer.eos_token_id, manifest["target_language_id"], tokenizer.unk_token_id]]
    )
    with torch.no_grad():
        expected = adapted(input_ids=source, decoder_input_ids=decoder).logits
        actual = merged(input_ids=source, decoder_input_ids=decoder).logits
    assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-5)
    assert merged.generation_config.forced_bos_token_id == manifest["target_language_id"]
    with pytest.raises(ValueError, match="new or empty"):
        merge_adapter(adapter_path, merged_path, local_files_only=True)


def test_reverse_direction_runs_via_cli(smoke_run):
    root, _, _ = smoke_run
    output = root / "reverse"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml.train.train_lora",
            "--dataset-dir",
            str(root / "dataset"),
            "--output-dir",
            str(output),
            "--direction",
            "ckb-en",
            "--smoke-test",
            "--limit",
            "200",
        ],
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads((output / "training_manifest.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(str(output / "adapter"), local_files_only=True)
    assert manifest["direction"] == "ckb-en"
    assert manifest["source_language_id"] == tokenizer.convert_tokens_to_ids("ckb_Arab")
    assert manifest["target_language_id"] == tokenizer.convert_tokens_to_ids("eng_Latn")


def test_merge_refuses_a_different_local_base(smoke_run, tmp_path):
    root, _, _ = smoke_run
    altered = tmp_path / "altered-base"
    shutil.copytree(root / "run" / "smoke_base", altered)
    config = altered / "config.json"
    config.write_text(config.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from the adapter"):
        merge_adapter(root / "run" / "adapter", tmp_path / "merged", base_model=str(altered))
    assert not (tmp_path / "merged").exists()


def test_distributed_writers_are_rejected_before_output_changes(smoke_run, monkeypatch):
    root, config, _ = smoke_run
    before = (root / "run" / "training_manifest.json").read_bytes()
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="one process"):
        run_training(config, smoke_test=True, resume="latest")
    assert (root / "run" / "training_manifest.json").read_bytes() == before


def test_training_input_output_and_checkpoint_guards(smoke_run, tmp_path):
    root, config, _ = smoke_run
    with pytest.raises(ValueError, match="must not contain one another"):
        run_training(replace(config, output_dir=str(root / "dataset")), smoke_test=True)
    link = tmp_path / "output-link"
    link.symlink_to(root / "run", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        run_training(replace(config, output_dir=str(link)), smoke_test=True)
    incomplete = root / "run" / "incomplete"
    incomplete.mkdir()
    with pytest.raises(ValueError, match="Checkpoint is incomplete"):
        run_training(config, smoke_test=True, resume=str(incomplete))


def test_validation_metric_drives_early_stopping(smoke_run, monkeypatch):
    root, config, _ = smoke_run
    original_smoke_config = train_lora._smoke_config

    def short_patience(settings):
        settings = original_smoke_config(settings)
        return replace(
            settings,
            training=replace(
                settings.training,
                epochs=2,
                eval_save_steps=2,
                early_stopping_patience=1,
            ),
        )

    monkeypatch.setattr(train_lora, "_smoke_config", short_patience)
    monkeypatch.setattr(
        train_lora.sacrebleu, "corpus_chrf", lambda *args, **kwargs: SimpleNamespace(score=20.0)
    )
    manifest = run_training(replace(config, output_dir=str(root / "early-stop")), smoke_test=True)
    assert manifest["best_chrf"] == 20.0
    assert manifest["global_step"] == 4  # No improvement on the second validation.


def test_phase_one_dataset_trains_only_with_explicit_unverified_override(tmp_path, local_films):
    dataset_output = tmp_path / "processed"
    build_dataset(
        BuildConfig(
            input_dir=local_films,
            output_dir=dataset_output,
            gold_path=tmp_path / "gold.jsonl",
            skip_language_id=True,
        )
    )
    config = replace(
        load_config(Path("ml/train/config.yaml")),
        dataset_dir=str(dataset_output / "dataset"),
        output_dir=str(tmp_path / "trained"),
    )
    with pytest.raises(ValueError, match="language-unverified"):
        run_training(config, smoke_test=True)
    assert not (tmp_path / "trained").exists()
    manifest = run_training(config, smoke_test=True, allow_unverified_data=True)
    assert manifest["train"]["samples"] == 8
    assert any("language-unverified" in warning for warning in manifest["warnings"])
    assert any("Only 8 training examples" in warning for warning in manifest["warnings"])


def test_cached_hub_revision_is_pinned_for_resume_and_merge(tmp_path):
    # A locally populated Hugging Face cache exercises Hub revision resolution
    # without downloading a model or pretending this tiny base is NLLB.
    rows = {name: _rows(name, count) for name, count in (("train", 6), ("val", 2), ("test", 2))}
    DatasetDict(rows).save_to_disk(str(tmp_path / "dataset"))
    examples = make_context_examples([dict(row) for row in rows["train"]], direction="en-ckb")
    create_smoke_base(examples, tmp_path / "base", seed=42)
    sha = "a" * 40
    repository = tmp_path / "hf" / "hub" / "models--example--tiny-translation"
    snapshot = repository / "snapshots" / sha
    snapshot.parent.mkdir(parents=True)
    shutil.copytree(tmp_path / "base", snapshot)
    (repository / "refs").mkdir()
    ref = repository / "refs" / "main"
    ref.write_text(sha, encoding="ascii")
    config = {
        "base_model": "example/tiny-translation",
        "dataset_dir": str(tmp_path / "dataset"),
        "output_dir": str(tmp_path / "run"),
        "local_files_only": True,
        "max_source_length": 32,
        "max_target_length": 16,
        "training": {
            "epochs": 1,
            "warmup_steps": 0,
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 4,
            "gradient_accumulation_steps": 2,
            "eval_save_steps": 1,
            "logging_steps": 1,
            "gradient_checkpointing": False,
            "mixed_precision": "fp32",
            "generation_num_beams": 1,
            "report_to": [],
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    env = {
        **os.environ,
        "HF_HOME": str(tmp_path / "hf"),
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
    }
    train_cmd = [sys.executable, "-m", "ml.train.train_lora", "--config", str(config_path)]
    first = subprocess.run(train_cmd, env=env, capture_output=True, text=True, timeout=120)
    assert first.returncode == 0, first.stderr
    manifest = json.loads((tmp_path / "run" / "training_manifest.json").read_text())
    assert manifest["base_revision_resolved"] == sha
    assert manifest["base_model_sha256"] is None
    # If resume or merge follows the moving 'main' ref, the missing new snapshot fails to load.
    ref.write_text("b" * 40, encoding="ascii")
    resumed = subprocess.run(
        [*train_cmd, "--resume-from-checkpoint", "latest"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert resumed.returncode == 0, resumed.stderr
    merged = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml.train.merge_adapter",
            "--adapter-dir",
            str(tmp_path / "run" / "adapter"),
            "--output-dir",
            str(tmp_path / "merged"),
            "--local-files-only",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert merged.returncode == 0, merged.stderr
    report = json.loads((tmp_path / "merged" / "merge_manifest.json").read_text())
    assert report["base_revision_resolved"] == sha
    assert report["base_identity_verified"] is True
