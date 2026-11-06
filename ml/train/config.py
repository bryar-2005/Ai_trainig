"""Strict configuration for reproducible, single-direction LoRA runs."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2")


@dataclass(frozen=True, slots=True)
class LoraSettings:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = DEFAULT_TARGETS


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    epochs: float = 3.0
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    warmup_steps: int = 500
    eval_save_steps: int = 200
    logging_steps: int = 20
    save_total_limit: int = 3
    early_stopping_patience: int = 3
    gradient_checkpointing: bool = True
    mixed_precision: str = "auto"
    generation_num_beams: int = 4
    seed: int = 42
    report_to: tuple[str, ...] = ("tensorboard",)


@dataclass(frozen=True, slots=True)
class RunConfig:
    base_model: str = "facebook/nllb-200-distilled-600M"
    base_revision: str | None = None
    direction: str = "en-ckb"
    dataset_dir: str = "data/processed/subtitles/dataset"
    output_dir: str = "outputs/nllb-600m-en-ckb-lora"
    context_sentences: int = 3
    max_source_length: int = 256
    max_target_length: int = 256
    local_files_only: bool = False
    lora: LoraSettings = field(default_factory=LoraSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


T = TypeVar("T")


def _parse_section(cls: type[T], values: Any, label: str) -> T:
    if not isinstance(values, dict):
        raise ValueError(f"{label} must be a YAML mapping")
    valid = {item.name for item in fields(cls)}
    extra = set(values) - valid
    if extra:
        raise ValueError(f"Unknown {label} option(s): {', '.join(sorted(map(str, extra)))}")
    values = dict(values)
    for name in ("target_modules", "report_to"):
        if name in values:
            if not isinstance(values[name], list) or not all(
                isinstance(item, str) and item for item in values[name]
            ):
                raise ValueError(f"{label}.{name} must be a list of nonempty strings")
            values[name] = tuple(values[name])
    return cls(**values)


def _int(value: Any, name: str, *, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _number(value: Any, name: str, *, minimum: float = 0.0, maximum: float | None = None) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value <= minimum
        or (maximum is not None and value > maximum)
    ):
        raise ValueError(f"{name} is outside its supported numeric range")


def validate_config(config: RunConfig) -> RunConfig:
    for name in ("base_model", "dataset_dir", "output_dir"):
        if not isinstance(getattr(config, name), str) or not getattr(config, name).strip():
            raise ValueError(f"{name} must be a nonempty string")
    if config.base_revision is not None and (
        not isinstance(config.base_revision, str) or not config.base_revision.strip()
    ):
        raise ValueError("base_revision must be a nonempty string or null")
    if not isinstance(config.direction, str) or config.direction not in {"en-ckb", "ckb-en"}:
        raise ValueError("direction must be en-ckb or ckb-en")
    _int(config.context_sentences, "context_sentences", minimum=3)
    if config.context_sentences > 5:
        raise ValueError("context_sentences must be between 3 and 5")
    for name in ("max_source_length", "max_target_length"):
        _int(getattr(config, name), name, minimum=8)
        if getattr(config, name) > 1024:
            raise ValueError(f"{name} must not exceed 1024")
    if not isinstance(config.local_files_only, bool):
        raise ValueError("local_files_only must be a boolean")
    _int(config.lora.r, "lora.r")
    _int(config.lora.alpha, "lora.alpha")
    if (
        isinstance(config.lora.dropout, bool)
        or not isinstance(config.lora.dropout, int | float)
        or not 0 <= config.lora.dropout < 1
    ):
        raise ValueError("lora.dropout must be in [0, 1)")
    if not config.lora.target_modules or len(set(config.lora.target_modules)) != len(
        config.lora.target_modules
    ):
        raise ValueError("lora.target_modules must contain distinct nonempty names")
    settings = config.training
    _number(settings.epochs, "training.epochs")
    _number(settings.learning_rate, "training.learning_rate")
    for name in (
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "eval_save_steps",
        "logging_steps",
        "save_total_limit",
        "early_stopping_patience",
        "generation_num_beams",
    ):
        _int(getattr(settings, name), f"training.{name}")
    _int(settings.warmup_steps, "training.warmup_steps", minimum=0)
    _int(settings.seed, "training.seed", minimum=0)
    if not isinstance(settings.gradient_checkpointing, bool):
        raise ValueError("training.gradient_checkpointing must be a boolean")
    if not isinstance(settings.mixed_precision, str) or settings.mixed_precision not in {
        "auto",
        "bf16",
        "fp16",
        "fp32",
    }:
        raise ValueError("training.mixed_precision must be auto, bf16, fp16, or fp32")
    if any(value not in {"tensorboard", "wandb"} for value in settings.report_to):
        raise ValueError("training.report_to accepts only tensorboard and wandb")
    if len(set(settings.report_to)) != len(settings.report_to):
        raise ValueError("training.report_to may not contain duplicates")
    return config


def load_config(path: str | Path) -> RunConfig:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot read training config {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("Training configuration must be a YAML mapping")
    raw = dict(raw)
    lora = _parse_section(LoraSettings, raw.pop("lora", {}), "lora")
    training = _parse_section(TrainingSettings, raw.pop("training", {}), "training")
    return validate_config(
        _parse_section(RunConfig, {**raw, "lora": lora, "training": training}, "run")
    )
