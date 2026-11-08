"""Lightweight hashing of local model and tokenizer assets."""

from __future__ import annotations

import hashlib
from pathlib import Path


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
