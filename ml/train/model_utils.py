"""Tokenizer preparation and a small, offline M2M100 model for smoke tests."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

from peft import LoraConfig, TaskType, get_peft_model
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import M2M100Config, M2M100ForConditionalGeneration, PreTrainedTokenizerFast

from ml.train.config import LoraSettings
from ml.train.dataset import CONTEXT_TOKEN, ContextExample


def add_context_token(tokenizer) -> int:
    """Register one real special token without replacing NLLB's language tokens."""
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [CONTEXT_TOKEN]}, replace_additional_special_tokens=False
    )
    token_id = tokenizer.convert_tokens_to_ids(CONTEXT_TOKEN)
    if token_id is None or token_id == tokenizer.unk_token_id:
        raise ValueError("Could not register the <ctx> special token")
    if token_id not in tokenizer.all_special_ids:
        raise ValueError("The <ctx> marker must be registered as a special token")
    return token_id


def resize_embeddings_for_tokenizer(model, tokenizer) -> None:
    """Extend embeddings if needed; never shrink a model's original vocabulary."""
    old_size = model.get_input_embeddings().num_embeddings
    if len(tokenizer) > old_size:
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)


def local_model_digest(source: str | Path) -> str | None:
    """Hash local base weights, configuration, and tokenizer assets for reproducibility."""
    directory = Path(source).expanduser()
    if not directory.is_dir():
        return None
    named = {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "merges.txt",
        "vocab.json",
    }
    files = sorted(
        entry
        for entry in directory.iterdir()
        if entry.is_file()
        and (
            entry.name in named
            or entry.name.endswith((".model", ".spm", ".safetensors"))
            or (
                entry.name.startswith(("model", "pytorch_model"))
                and entry.name.endswith((".bin", ".index.json"))
            )
        )
    )
    if not files:
        raise ValueError(f"Local base model directory has no recognized model assets: {directory}")
    digest = hashlib.sha256()
    for entry in files:
        digest.update(entry.name.encode("utf-8") + b"\0")
        with entry.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def attach_lora(model, settings: LoraSettings, context_token_id: int):
    """Train LoRA projections plus only the new sentinel embedding row.

    PEFT handles the M2M100 shared encoder/decoder/head weights. Saving is done
    with ``save_embedding_layers=False``; the selected trainable token row is
    contained in the adapter, avoiding a copy of the enormous NLLB vocabulary.
    """
    model.config.use_cache = False
    config = LoraConfig(
        r=settings.r,
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        target_modules=list(settings.target_modules),
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
        trainable_token_indices=[context_token_id],
    )
    result = get_peft_model(model, config)
    token_parameters = [
        parameter
        for name, parameter in result.named_parameters()
        if "trainable_tokens_delta" in name and parameter.requires_grad
    ]
    if not token_parameters:
        raise ValueError("PEFT did not create a trainable <ctx> embedding")
    return result


def create_smoke_base(
    examples: Iterable[ContextExample], destination: Path, *, seed: int
) -> tuple[PreTrainedTokenizerFast, M2M100ForConditionalGeneration]:
    """Create a tiny random M2M100 checkpoint, solely for offline wiring tests.

    This model has never learned translation and is not a substitute for the
    real NLLB checkpoint. It deliberately adds <ctx> *after* model creation so
    the smoke test also covers the real resized-embedding adapter workflow.
    """
    import torch

    samples = list(examples)
    if not samples:
        raise ValueError("Cannot build a smoke tokenizer from an empty training set")
    backend = Tokenizer(models.WordLevel(unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(
        vocab_size=2048,
        special_tokens=["<pad>", "<unk>", "</s>", "<s>", "eng_Latn", "ckb_Arab"],
    )
    backend.train_from_iterator(
        (
            text
            for example in samples
            for text in (example.source, example.target, *example.context)
        ),
        trainer=trainer,
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        unk_token="<unk>",
        eos_token="</s>",
        bos_token="<s>",
        additional_special_tokens=["eng_Latn", "ckb_Arab"],
        model_max_length=256,
    )
    original_vocab_size = len(tokenizer)
    add_context_token(tokenizer)
    torch.manual_seed(seed)
    config = M2M100Config(
        vocab_size=original_vocab_size,
        d_model=32,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        max_position_embeddings=256,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        decoder_start_token_id=tokenizer.eos_token_id,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
    )
    model = M2M100ForConditionalGeneration(config)
    destination.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(destination, safe_serialization=True)
    tokenizer.save_pretrained(destination)
    model.config._name_or_path = str(destination)
    return tokenizer, model
