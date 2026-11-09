"""Fine-tune a context-aware English–Sorani NLLB LoRA adapter.

The smoke mode runs the same data, LoRA, evaluation, checkpoint, and adapter
paths with a tiny randomly initialized M2M100 model; it does not evaluate or
train the real NLLB checkpoint. Use a real authorized dataset in either mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import sacrebleu
import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, load_from_disk
from peft import PeftModel
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    set_seed,
)
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.trainer_utils import get_last_checkpoint

from ml.train.checkpoint import restore_m2m100_tied_weights
from ml.train.config import RunConfig, load_config, validate_config
from ml.train.dataset import (
    ContextExample,
    dataset_fingerprint,
    make_context_examples,
    token_ids,
    tokenize_example,
    validate_splits,
)
from ml.train.model_utils import (
    add_context_token,
    attach_lora,
    create_smoke_base,
    local_model_digest,
    resize_embeddings_for_tokenizer,
)

_MANIFEST = "training_manifest.json"


class WeightedCollator:
    """Pad only model fields; carry provenance weights separately."""

    def __init__(self, tokenizer, model) -> None:
        self.collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, label_pad_token_id=-100, return_tensors="pt"
        )

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        weights = torch.tensor([float(row["sample_weight"]) for row in rows], dtype=torch.float32)
        features = [
            {key: row[key] for key in ("input_ids", "attention_mask", "labels")} for row in rows
        ]
        batch = self.collator(features)
        batch["sample_weight"] = weights
        return batch


class WeightedSeq2SeqTrainer(Seq2SeqTrainer):
    """Per-example provenance-weighted token cross-entropy and compact PEFT saves."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Our custom loss uses num_items_in_batch to normalize across an entire
        # accumulation window, even when the last microbatch is smaller.
        self.model_accepts_loss_kwargs = True

    def get_batch_samples(self, epoch_iterator, num_batches, device):
        batches = []
        for _ in range(num_batches):
            try:
                batches.append(next(epoch_iterator))
            except StopIteration:
                break
        count = sum(batch["labels"].shape[0] for batch in batches)
        # Unlike Trainer's default token count, provenance weighting is per
        # example. run_training explicitly restricts execution to one process.
        return batches, count if batches else None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        weights = inputs.get("sample_weight")
        payload = {
            key: value for key, value in inputs.items() if key not in {"labels", "sample_weight"}
        }
        if "decoder_input_ids" not in payload:
            # M2M100/NLLB does not implement prepare_decoder_input_ids_from_labels.
            # Match its standard teacher-forcing shift without asking the model to
            # calculate a second, unweighted cross-entropy loss.
            start_id = model.config.decoder_start_token_id
            pad_id = model.config.pad_token_id
            if start_id is None or pad_id is None:
                raise ValueError("The model must define decoder start and padding token IDs")
            shifted = labels.new_full(labels.shape, pad_id)
            shifted[:, 1:] = labels[:, :-1].clone()
            shifted[:, 0] = start_id
            shifted.masked_fill_(shifted == -100, pad_id)
            payload["decoder_input_ids"] = shifted
        outputs = model(**payload)
        logits = outputs.logits
        per_token = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape(labels.shape)
        mask = labels.ne(-100)
        token_counts = mask.sum(dim=1)
        if torch.any(token_counts == 0):
            raise ValueError("A training example has no target tokens")
        per_sample = (per_token * mask).sum(dim=1) / token_counts
        if weights is None:
            weights = torch.ones_like(per_sample)
        weighted = per_sample * weights.to(per_sample.device)
        if num_items_in_batch is None:
            weighted_loss = weighted.mean()
        elif num_items_in_batch <= 0:
            raise ValueError("The accumulation window must contain training examples")
        else:
            weighted_loss = weighted.sum() / num_items_in_batch
        return (weighted_loss, outputs) if return_outputs else weighted_loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None, **gen_kwargs):
        # Seq2SeqTrainer passes all batch fields to generate() and directly to
        # model.forward() during evaluation. The weight is training-only; eval
        # loss is unweighted and chrF++ is measured on the actual translations.
        clean_inputs = {key: value for key, value in inputs.items() if key != "sample_weight"}
        return super().prediction_step(
            model, clean_inputs, prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )

    def _save(self, output_dir: str | None = None, state_dict=None) -> None:
        """Persist adapter and trainable token row, not the entire resized embedding."""
        directory = Path(output_dir or self.args.output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        model = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
        if not isinstance(model, PeftModel):
            raise ValueError("Expected a PEFT model when saving a LoRA checkpoint")
        model.save_pretrained(
            str(directory),
            state_dict=state_dict,
            safe_serialization=True,
            save_embedding_layers=False,
        )
        if self.processing_class is not None:
            self.processing_class.save_pretrained(str(directory))
        torch.save(self.args, directory / TRAINING_ARGS_NAME)


def _load_records(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.is_dir():
        raise ValueError(f"DatasetDict directory does not exist: {path}")
    try:
        dataset = load_from_disk(str(path))
    except (OSError, ValueError, FileNotFoundError) as exc:
        raise ValueError(f"Could not load DatasetDict from {path}: {exc}") from exc
    if not isinstance(dataset, DatasetDict) or not {"train", "val", "test"}.issubset(dataset):
        raise ValueError("Expected a DatasetDict with train, val and test splits from Phase 1")
    return {name: [dict(row) for row in dataset[name]] for name in ("train", "val", "test")}


def _smoke_config(config: RunConfig) -> RunConfig:
    return replace(
        config,
        max_source_length=min(config.max_source_length, 64),
        max_target_length=min(config.max_target_length, 32),
        training=replace(
            config.training,
            epochs=1,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=1,
            warmup_steps=0,
            eval_save_steps=5,
            logging_steps=1,
            mixed_precision="fp32",
            generation_num_beams=1,
            report_to=(),
        ),
    )


def _precision(mode: str) -> tuple[bool, bool, str]:
    cuda = torch.cuda.is_available()
    if mode == "auto":
        mode = "bf16" if cuda and torch.cuda.is_bf16_supported() else "fp16" if cuda else "fp32"
    if mode in {"bf16", "fp16"} and not cuda:
        raise ValueError(f"{mode} requires a CUDA GPU in this training configuration; choose fp32")
    if mode == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("The CUDA GPU does not support bf16; choose fp16 or auto")
    return mode == "bf16", mode == "fp16", mode


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=".manifest-", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _output_paths(
    config: RunConfig, dataset_path: Path, resume: str | None
) -> tuple[Path, Path | None]:
    requested = Path(config.output_dir).expanduser()
    if requested.is_symlink():
        raise ValueError("Training output may not be a symlink")
    output = requested.resolve()
    if (
        output == dataset_path
        or output.is_relative_to(dataset_path)
        or dataset_path.is_relative_to(output)
    ):
        raise ValueError("Training output and input dataset must not contain one another")
    if resume is None:
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError(
                "Training output is not empty; choose another directory or resume a checkpoint"
            )
        output.mkdir(parents=True, exist_ok=True)
        return output, None
    if not output.is_dir() or not (output / _MANIFEST).is_file():
        raise ValueError("Cannot resume: output does not contain a training manifest")
    checkpoint = get_last_checkpoint(str(output)) if resume == "latest" else resume
    if checkpoint is None:
        raise ValueError("No checkpoint found to resume")
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_relative_to(output) or not checkpoint_path.is_dir():
        raise ValueError("The checkpoint must be a directory inside this run's output directory")
    required = {
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "adapter_config.json",
        "adapter_model.safetensors",
    }
    missing = sorted(name for name in required if not (checkpoint_path / name).is_file())
    if not (checkpoint_path / "rng_state.pth").is_file() and not list(
        checkpoint_path.glob("rng_state_*.pth")
    ):
        missing.append("rng_state.pth")
    if missing:
        raise ValueError(f"Checkpoint is incomplete; missing {', '.join(missing)}")
    return output, checkpoint_path


def _metrics(tokenizer):
    def compute(prediction):
        predictions = (
            prediction.predictions[0]
            if isinstance(prediction.predictions, tuple)
            else prediction.predictions
        )
        labels = np.where(
            prediction.label_ids == -100, tokenizer.pad_token_id, prediction.label_ids
        )
        hypotheses = [
            item.strip() for item in tokenizer.batch_decode(predictions, skip_special_tokens=True)
        ]
        references = [
            item.strip() for item in tokenizer.batch_decode(labels, skip_special_tokens=True)
        ]
        score = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2).score
        return {"chrf": float(score)}

    return compute


def _tokenize_rows(
    examples: list[ContextExample], tokenizer, config: RunConfig
) -> tuple[Dataset, dict[str, int]]:
    records = [
        tokenize_example(
            item,
            tokenizer,
            direction=config.direction,
            max_source_length=config.max_source_length,
            max_target_length=config.max_target_length,
        )
        for item in examples
    ]
    diagnostics = {
        "samples": len(records),
        "with_previous_context": sum(item["context_sentences_available"] > 0 for item in records),
        "context_truncated": sum(item["context_truncated"] for item in records),
        "source_truncated": sum(item["source_truncated"] for item in records),
        "target_truncated": sum(item["target_truncated"] for item in records),
    }
    return Dataset.from_list(records), diagnostics


def _training_args(
    config: RunConfig, *, train_samples: int, smoke: bool
) -> tuple[Seq2SeqTrainingArguments, str]:
    settings = config.training
    bf16, fp16, precision = _precision(settings.mixed_precision)
    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    updates_per_epoch = math.ceil(
        train_samples
        / (settings.per_device_train_batch_size * settings.gradient_accumulation_steps * world_size)
    )
    total_updates = max(1, math.ceil(updates_per_epoch * settings.epochs))
    save_steps = min(settings.eval_save_steps, total_updates)
    return (
        Seq2SeqTrainingArguments(
            output_dir=config.output_dir,
            overwrite_output_dir=False,
            do_train=True,
            do_eval=True,
            num_train_epochs=settings.epochs,
            per_device_train_batch_size=settings.per_device_train_batch_size,
            per_device_eval_batch_size=settings.per_device_eval_batch_size,
            gradient_accumulation_steps=settings.gradient_accumulation_steps,
            learning_rate=settings.learning_rate,
            lr_scheduler_type="cosine",
            warmup_steps=settings.warmup_steps,
            gradient_checkpointing=settings.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            bf16=bf16,
            fp16=fp16,
            eval_strategy="steps",
            save_strategy="steps",
            eval_steps=save_steps,
            save_steps=save_steps,
            logging_strategy="steps",
            logging_steps=min(settings.logging_steps, total_updates),
            save_total_limit=settings.save_total_limit,
            load_best_model_at_end=True,
            restore_callback_states_from_checkpoint=True,
            metric_for_best_model="chrf",
            greater_is_better=True,
            predict_with_generate=True,
            generation_max_length=config.max_target_length,
            generation_num_beams=settings.generation_num_beams,
            remove_unused_columns=False,
            report_to=list(settings.report_to),
            save_safetensors=True,
            seed=settings.seed,
            data_seed=settings.seed,
            dataloader_num_workers=0,
            disable_tqdm=smoke,
            optim="adamw_torch",
        ),
        precision,
    )


class TrainingProgress(TrainerCallback):
    """Report optimizer steps; raising in the callback aborts the run safely."""

    def __init__(self, callback: Callable[[dict[str, Any]], None]):
        self.callback = callback

    def on_step_end(self, args, state, control, **kwargs):
        self.callback(
            {"stage": "training", "step": state.global_step, "total_steps": state.max_steps}
        )

    def on_prediction_step(self, args, state, control, **kwargs):
        self.callback(
            {"stage": "validation", "step": state.global_step, "total_steps": state.max_steps}
        )


def run_training(
    config: RunConfig,
    *,
    smoke_test: bool = False,
    limit: int = 200,
    resume: str | None = None,
    allow_unverified_data: bool = False,
    initial_adapter: str | Path | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Train one direction, checkpoint full state, and save the best adapter."""
    config = validate_config(_smoke_config(config) if smoke_test else config)
    adapter_path: Path | None = None
    adapter_manifest: dict[str, Any] = {}
    adapter_digest: str | None = None
    if initial_adapter is not None:
        if smoke_test:
            raise ValueError(
                "An initial adapter cannot be combined with a newly randomized smoke base"
            )
        requested = Path(initial_adapter).expanduser()
        if requested.is_symlink():
            raise ValueError("Initial adapter may not be a symlink")
        adapter_path = requested.resolve()
        required = ("adapter_config.json", "adapter_model.safetensors", "training_manifest.json")
        if not adapter_path.is_dir() or any(
            not (adapter_path / name).is_file() for name in required
        ):
            raise ValueError("Initial adapter is missing a completed Phase 2 model or manifest")
        try:
            adapter_manifest = json.loads(
                (adapter_path / "training_manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("Cannot read initial adapter manifest") from exc
        if not isinstance(adapter_manifest, dict):
            raise ValueError("Initial adapter manifest must be an object")
        previous_config = adapter_manifest.get("config")
        if adapter_manifest.get("status") != "completed" or not isinstance(previous_config, dict):
            raise ValueError("Initial adapter manifest is incomplete")
        if previous_config.get("direction") != config.direction:
            raise ValueError(
                "Initial adapter direction differs from the requested training direction"
            )
        expected_lora = previous_config.get("lora")
        actual_lora = config.as_dict()["lora"]
        if json.dumps(expected_lora, sort_keys=True) != json.dumps(actual_lora, sort_keys=True):
            raise ValueError("LoRA settings must match the initial adapter")
        expected_base = adapter_manifest.get("base_model_sha256")
        if expected_base is not None:
            if local_model_digest(config.base_model) != expected_base:
                raise ValueError("Initial adapter's local base hash does not match")
        elif (
            config.base_model != adapter_manifest.get("actual_base_model")
            or not adapter_manifest.get("base_revision_resolved")
            or (
                config.base_revision
                and config.base_revision != adapter_manifest["base_revision_resolved"]
            )
        ):
            raise ValueError("Initial adapter's Hub base revision cannot be verified")
        digest = hashlib.sha256()
        for path in sorted(adapter_path.iterdir()):
            if path.is_file() and path.suffix in {".json", ".safetensors", ".model", ".spm"}:
                digest.update(path.name.encode() + b"\\0")
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
        adapter_digest = digest.hexdigest()
        output_path = Path(config.output_dir).expanduser().resolve()
        if (
            output_path == adapter_path
            or output_path.is_relative_to(adapter_path)
            or adapter_path.is_relative_to(output_path)
        ):
            raise ValueError("Training output and initial adapter must not contain one another")
    if not 200 <= limit <= 500:
        raise ValueError("--limit must be between 200 and 500")
    # A run owns one output directory and normalizes examples across one
    # accumulation window. Do not silently launch untested distributed writers.
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or (
        torch.cuda.is_available() and torch.cuda.device_count() > 1
    ):
        raise ValueError(
            "Training supports one process and at most one visible GPU; "
            "set CUDA_VISIBLE_DEVICES to select a single GPU"
        )
    dataset_path = Path(config.dataset_dir).expanduser().resolve()
    splits = _load_records(dataset_path)
    validate_splits(splits, allow_unverified=allow_unverified_data)
    fingerprint = dataset_fingerprint(splits)
    training_examples = make_context_examples(
        splits["train"], direction=config.direction, context_sentences=config.context_sentences
    )
    validation_examples = make_context_examples(
        splits["val"], direction=config.direction, context_sentences=config.context_sentences
    )
    if smoke_test:
        training_examples = training_examples[:limit]
        validation_examples = validation_examples[:50]
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    identity = {
        "schema_version": 1,
        "config": config.as_dict(),
        "dataset_sha256": fingerprint,
        "smoke_test": smoke_test,
        "smoke_limit": limit if smoke_test else None,
        "allow_unverified_data": allow_unverified_data,
    }
    # Preserve the pre-existing resume identity for runs without an initial adapter.
    if adapter_path is not None:
        identity["initial_adapter_sha256"] = adapter_digest
    identity_hash = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output, checkpoint = _output_paths(config, dataset_path, resume)
    manifest_path = output / _MANIFEST
    previous: dict[str, Any] = {}
    if checkpoint:
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read training manifest: {exc}") from exc
        if previous.get("identity_sha256") != identity_hash:
            raise ValueError("Resume configuration or dataset differs from the original run")
    manifest: dict[str, Any] = {
        **identity,
        "identity_sha256": identity_hash,
        "direction": config.direction,
        "status": "initializing",
        "warnings": [],
    }
    if adapter_path is not None:
        manifest["initial_adapter"] = str(adapter_path)
        manifest["initial_training_identity_sha256"] = adapter_manifest.get("identity_sha256")
        if adapter_manifest.get("smoke_test"):
            manifest["smoke_test"] = True
            manifest["warnings"].append(
                "The initial adapter uses a random smoke base and cannot translate meaningfully."
            )
    if smoke_test:
        manifest["warnings"].append(
            "Smoke model is random and cannot translate; no NLLB weights were trained."
        )
        if len(training_examples) < limit:
            manifest["warnings"].append(
                f"Only {len(training_examples)} training examples are available; "
                f"requested limit was {limit}."
            )
    if allow_unverified_data:
        unverified_count = sum(
            not row["language_verified"] for group in splits.values() for row in group
        )
        if unverified_count:
            manifest["warnings"].append(
                f"The dataset contains {unverified_count} language-unverified pairs."
            )
    _write_json(manifest_path, manifest)
    try:
        set_seed(config.training.seed)
        smoke_base = output / "smoke_base"
        if smoke_test and not checkpoint:
            tokenizer, base_model = create_smoke_base(
                training_examples, smoke_base, seed=config.training.seed
            )
            actual_base = str(smoke_base)
            base_digest = local_model_digest(actual_base)
            resolved_revision = None
        else:
            actual_base = str(smoke_base) if smoke_test else config.base_model
            base_digest = local_model_digest(actual_base)
            if checkpoint and previous.get("base_model_sha256") != base_digest:
                raise ValueError("The local base model differs from the checkpoint's original base")
            requested_revision = (
                previous.get("base_revision_resolved")
                or adapter_manifest.get("base_revision_resolved")
                or config.base_revision
            )
            if checkpoint and base_digest is None and not requested_revision:
                raise ValueError(
                    "The checkpoint does not record a reproducible base model revision"
                )
            from ml.inference.model_source import resolve_model_source

            local_only = smoke_test or config.local_files_only
            load_source, cached_revision = resolve_model_source(
                actual_base, revision=requested_revision, local_files_only=local_only
            )
            model_kwargs: dict[str, Any] = {}
            if requested_revision and load_source == actual_base and base_digest is None:
                model_kwargs["adapter_kwargs"] = {"revision": requested_revision}
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                load_source,
                revision=requested_revision,
                local_files_only=local_only,
                trust_remote_code=False,
                **model_kwargs,
            )
            restore_m2m100_tied_weights(base_model, load_source, revision=requested_revision)
            resolved_revision = (
                None
                if base_digest is not None
                else (getattr(base_model.config, "_commit_hash", None) or cached_revision)
            )
            if base_digest is None and not resolved_revision:
                # A full Hub commit hash supplied by the user is already immutable.
                if (
                    requested_revision
                    and len(requested_revision) == 40
                    and all(char in "0123456789abcdef" for char in requested_revision.lower())
                ):
                    resolved_revision = requested_revision
                else:
                    raise ValueError(
                        "Could not resolve the base model's Hub commit; set base_revision "
                        "to an immutable commit hash"
                    )
            if checkpoint and previous.get("base_revision_resolved") != resolved_revision:
                raise ValueError("The resolved base model revision differs from the checkpoint")
            if (
                adapter_manifest.get("base_revision_resolved")
                and adapter_manifest["base_revision_resolved"] != resolved_revision
            ):
                raise ValueError(
                    "The resolved base model revision differs from the initial adapter"
                )
            tokenizer = AutoTokenizer.from_pretrained(
                str(adapter_path) if adapter_path else load_source,
                revision=None if adapter_path else (resolved_revision or requested_revision),
                local_files_only=local_only,
                trust_remote_code=False,
            )
        context_token_id = add_context_token(tokenizer)
        resize_embeddings_for_tokenizer(base_model, tokenizer)
        source_id, target_id, _, _, _ = token_ids(tokenizer, config.direction)
        if adapter_path and (
            context_token_id != adapter_manifest.get("context_token_id")
            or target_id != adapter_manifest.get("target_language_id")
        ):
            raise ValueError("Initial adapter tokenizer IDs differ from its manifest")
        base_model.generation_config.forced_bos_token_id = target_id
        if adapter_path:
            base_model.config.use_cache = False
            model = PeftModel.from_pretrained(
                base_model, str(adapter_path), is_trainable=True, local_files_only=True
            )
            if not any(p.requires_grad for p in model.parameters()):
                raise ValueError("Initial adapter did not expose trainable parameters")
        else:
            model = attach_lora(base_model, config.lora, context_token_id)
        train_data, train_stats = _tokenize_rows(training_examples, tokenizer, config)
        val_data, val_stats = _tokenize_rows(validation_examples, tokenizer, config)
        args, precision = _training_args(config, train_samples=len(train_data), smoke=smoke_test)
        if args.warmup_steps >= max(
            1,
            math.ceil(
                len(train_data)
                / (args.per_device_train_batch_size * args.gradient_accumulation_steps)
            )
            * args.num_train_epochs,
        ):
            manifest["warnings"].append(
                "Warmup covers the full run; consider lowering warmup_steps for small datasets."
            )
        manifest.update(
            {
                "status": "training",
                "actual_base_model": actual_base,
                "base_model_sha256": base_digest,
                "base_revision_resolved": resolved_revision,
                "precision": precision,
                "context_token_id": context_token_id,
                "source_language_id": source_id,
                "target_language_id": target_id,
                "train": train_stats,
                "validation": val_stats,
                "train_provenance": dict(
                    sorted(Counter(item.provenance for item in training_examples).items())
                ),
                "eval_save_steps_effective": args.eval_steps,
                "trainable_parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "total_parameters": sum(p.numel() for p in model.parameters()),
            }
        )
        _write_json(manifest_path, manifest)
        trainer = WeightedSeq2SeqTrainer(
            model=model,
            args=args,
            train_dataset=train_data,
            eval_dataset=val_data,
            data_collator=WeightedCollator(tokenizer, model),
            processing_class=tokenizer,
            compute_metrics=_metrics(tokenizer),
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=config.training.early_stopping_patience
                ),
                *([TrainingProgress(progress)] if progress is not None else []),
            ],
        )
        result = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
        evaluation = trainer.evaluate()
        trainer.model.config.use_cache = True
        adapter_dir = output / "adapter"
        trainer.save_model(str(adapter_dir))
        manifest.update(
            {
                "status": "completed",
                "global_step": trainer.state.global_step,
                "best_checkpoint": trainer.state.best_model_checkpoint,
                "best_chrf": trainer.state.best_metric,
                "final_eval_chrf": evaluation.get("eval_chrf"),
                "training_runtime_seconds": result.metrics.get("train_runtime"),
                "adapter_dir": str(adapter_dir),
            }
        )
        _write_json(manifest_path, manifest)
        _write_json(adapter_dir / _MANIFEST, manifest)
        return manifest
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error_type"] = type(exc).__name__
        _write_json(manifest_path, manifest)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--direction", choices=["en-ckb", "ckb-en"])
    parser.add_argument(
        "--smoke-test", action="store_true", help="Use a tiny random offline M2M100 model"
    )
    parser.add_argument(
        "--limit", type=int, default=200, help="Smoke training examples to keep (200–500)"
    )
    parser.add_argument(
        "--resume-from-checkpoint", help="Checkpoint inside output directory, or 'latest'"
    )
    parser.add_argument("--allow-unverified-data", action="store_true")
    parser.add_argument(
        "--initial-adapter", type=Path, help="Continue a completed compatible LoRA adapter"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config)
        overrides = {}
        if args.dataset_dir is not None:
            overrides["dataset_dir"] = str(args.dataset_dir)
        if args.output_dir is not None:
            overrides["output_dir"] = str(args.output_dir)
        if args.direction is not None:
            overrides["direction"] = args.direction
        config = replace(config, **overrides)
        manifest = run_training(
            config,
            smoke_test=args.smoke_test,
            limit=args.limit,
            resume=args.resume_from_checkpoint,
            allow_unverified_data=args.allow_unverified_data,
            initial_adapter=args.initial_adapter,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                key: manifest.get(key)
                for key in (
                    "status",
                    "smoke_test",
                    "direction",
                    "global_step",
                    "best_chrf",
                    "adapter_dir",
                    "warnings",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
