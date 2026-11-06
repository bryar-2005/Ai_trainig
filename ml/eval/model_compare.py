"""Generate baseline and adapter hypotheses using the same held-out context."""

from __future__ import annotations

import json
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed

from ml.eval.records import EvaluationRecord, file_sha256
from ml.train.dataset import token_ids, tokenize_example
from ml.train.model_utils import local_model_digest, resize_embeddings_for_tokenizer


def _adapter_manifest(adapter: Path) -> dict[str, Any]:
    path = adapter / "training_manifest.json"
    if not path.is_file() or not (adapter / "adapter_model.safetensors").is_file():
        raise ValueError("Adapter must contain its training manifest and safetensors weights")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read adapter manifest: {exc}") from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("config"), dict):
        raise ValueError("Invalid adapter training manifest")
    if manifest.get("status") != "completed":
        raise ValueError("Only a completed training run's adapter can be evaluated")
    return manifest


def generate_comparison(
    records: list[EvaluationRecord],
    *,
    adapter_dir: Path,
    direction: str,
    dataset_sha256: str,
    device: str = "cpu",
    batch_size: int = 4,
    base_model_override: str | None = None,
    local_files_only: bool = False,
    allow_smoke_model: bool = False,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Run unadapted base first, then the saved adapter, using identical inputs.

    The zero-shot base sees the added <ctx> delimiter without the adapter's
    trained row. That row may be newly random or previously unused padding;
    it is not necessarily a delimiter learned by the pretrained model.
    """
    if device not in {"cpu", "cuda"} or batch_size < 1:
        raise ValueError("Invalid inference device or batch size")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    adapter = adapter_dir.expanduser().resolve()
    manifest = _adapter_manifest(adapter)
    config = manifest["config"]
    if config.get("direction") != direction:
        raise ValueError("Evaluation direction does not match the adapter")
    if manifest.get("dataset_sha256") != dataset_sha256:
        raise ValueError(
            "Evaluation dataset differs from the adapter's film-disjoint training dataset"
        )
    if manifest.get("smoke_test") and not allow_smoke_model:
        raise ValueError(
            "The adapter is a random smoke model; pass --allow-smoke-model only for tests"
        )
    source = base_model_override or manifest.get("actual_base_model")
    if not isinstance(source, str) or not source:
        raise ValueError("Adapter manifest does not identify a base model")
    if Path(source).expanduser().is_dir():
        source = str(Path(source).expanduser().resolve())
    expected_digest = manifest.get("base_model_sha256")
    digest = local_model_digest(source)
    if expected_digest is not None:
        if digest != expected_digest:
            raise ValueError("Base model does not match the adapter's recorded local base hash")
        revision = None
    else:
        revision = manifest.get("base_revision_resolved")
        if not revision or source != manifest.get("actual_base_model"):
            raise ValueError("Cannot verify an overridden Hub base against the adapter")
    training = config.get("training")
    if not isinstance(training, dict):
        raise ValueError("Adapter manifest is missing its training settings")
    try:
        context_window = int(config["context_sentences"])
        source_limit = int(config["max_source_length"])
        target_limit = int(config["max_target_length"])
        beams = int(training["generation_num_beams"])
        seed = int(training["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Adapter manifest has invalid generation settings") from exc
    if not 3 <= context_window <= 5 or min(source_limit, target_limit, beams) < 1:
        raise ValueError("Adapter manifest has unsupported generation settings")
    set_seed(seed)
    base = AutoModelForSeq2SeqLM.from_pretrained(
        source,
        revision=revision,
        local_files_only=local_files_only or bool(manifest.get("smoke_test")),
    )
    loaded_revision = getattr(base.config, "_commit_hash", None)
    if revision and loaded_revision and loaded_revision != revision:
        raise ValueError("Loaded base revision does not match the adapter")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter), local_files_only=True)
    _, target_id, context_id, _, _ = token_ids(tokenizer, direction)
    if context_id != manifest.get("context_token_id") or target_id != manifest.get(
        "target_language_id"
    ):
        raise ValueError("Tokenizer IDs do not match the adapter manifest")
    resize_embeddings_for_tokenizer(base, tokenizer)
    base.config.use_cache = True
    base.generation_config.forced_bos_token_id = target_id
    base.to(device).eval()
    encoded = [
        tokenize_example(
            replace(record.example, context=record.example.context[-context_window:]),
            tokenizer,
            direction=direction,
            max_source_length=source_limit,
            max_target_length=target_limit,
        )
        for record in records
    ]

    def predict(model) -> list[str]:
        predictions: list[str] = []
        for start in range(0, len(encoded), batch_size):
            features = [
                {key: row[key] for key in ("input_ids", "attention_mask")}
                for row in encoded[start : start + batch_size]
            ]
            inputs = tokenizer.pad(features, padding=True, return_tensors="pt").to(device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    forced_bos_token_id=target_id,
                    max_length=target_limit,
                    num_beams=beams,
                )
            predictions.extend(
                text.strip() for text in tokenizer.batch_decode(generated, skip_special_tokens=True)
            )
        return predictions

    baseline = predict(base)
    adapted = PeftModel.from_pretrained(base, str(adapter), is_trainable=False).to(device).eval()
    candidate = predict(adapted)
    info = {
        "mode": "generated",
        "base_model": source,
        "base_model_sha256": digest,
        "base_revision_resolved": revision,
        "adapter_sha256": file_sha256(adapter / "adapter_model.safetensors"),
        "adapter_config_sha256": file_sha256(adapter / "adapter_config.json"),
        "adapter_assets_sha256": local_model_digest(adapter),
        "training_manifest_sha256": file_sha256(adapter / "training_manifest.json"),
        "training_identity_sha256": manifest.get("identity_sha256"),
        "smoke_model": bool(manifest.get("smoke_test")),
        "context_window": context_window,
        "examples_with_context": sum(item["context_sentences_used"] > 0 for item in encoded),
        "context_truncated": sum(item["context_truncated"] for item in encoded),
        "source_truncated": sum(item["source_truncated"] for item in encoded),
        "max_source_length": source_limit,
        "max_target_length": target_limit,
        "num_beams": beams,
        "seed": seed,
        "libraries": {
            "torch": str(torch.__version__),
            "transformers": version("transformers"),
            "peft": version("peft"),
        },
        "baseline_context_caveat": (
            "The zero-shot base receives the same context format without the adapter's trained "
            "<ctx> row. The base row may be randomly initialized or unused, and is not necessarily "
            "a pretrained context delimiter."
        ),
    }
    return baseline, candidate, info
