"""Merge a trained PEFT adapter and its trainable context token into its base."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from ml.train.dataset import token_ids
from ml.train.model_utils import local_model_digest, resize_embeddings_for_tokenizer


def _manifest(adapter: Path) -> dict[str, Any]:
    for candidate in (
        adapter / "training_manifest.json",
        adapter.parent / "training_manifest.json",
    ):
        if candidate.is_file():
            try:
                value = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Could not read {candidate}: {exc}") from exc
            if not isinstance(value, dict) or not isinstance(value.get("config"), dict):
                raise ValueError(f"Invalid training manifest: {candidate}")
            return value
    raise ValueError("No training manifest found next to or inside the adapter")


def merge_adapter(
    adapter_dir: str | Path,
    output_dir: str | Path,
    *,
    base_model: str | None = None,
    local_files_only: bool = False,
    device: str = "cpu",
) -> dict[str, Any]:
    """Save a standalone model, tokenizer, and merge provenance into a new directory."""
    source = Path(adapter_dir).expanduser()
    target = Path(output_dir).expanduser()
    if source.is_symlink() or target.is_symlink():
        raise ValueError("Adapter and merge output paths may not be symlinks")
    source, target = source.resolve(), target.resolve()
    if not source.is_dir() or not (source / "adapter_config.json").is_file():
        raise ValueError(f"Not a PEFT adapter directory: {source}")
    if source == target or source.is_relative_to(target) or target.is_relative_to(source):
        raise ValueError("Merge output and adapter directories must not contain one another")
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError(f"Merge output must be a new or empty directory: {target}")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    manifest = _manifest(source)
    direction = manifest["config"].get("direction")
    model_source = base_model or manifest.get("actual_base_model")
    if not isinstance(model_source, str) or not model_source:
        raise ValueError("The manifest has no base model; pass --base-model explicitly")
    tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True)
    _, target_id, context_id, _, _ = token_ids(tokenizer, direction)
    if context_id != manifest.get("context_token_id") or target_id != manifest.get(
        "target_language_id"
    ):
        raise ValueError("Tokenizer language/context IDs do not match the training manifest")
    base_digest = local_model_digest(model_source)
    expected_digest = manifest.get("base_model_sha256")
    if expected_digest is not None and base_digest != expected_digest:
        raise ValueError("The local base model differs from the adapter's original base")
    recorded_source = manifest.get("actual_base_model")
    revision = manifest.get("base_revision_resolved") if model_source == recorded_source else None
    base_identity_verified = bool(
        (expected_digest is not None and base_digest == expected_digest)
        or (model_source == recorded_source and revision)
    )
    base = AutoModelForSeq2SeqLM.from_pretrained(
        model_source, revision=revision, local_files_only=local_files_only
    )
    loaded_revision = getattr(base.config, "_commit_hash", None)
    if revision and loaded_revision and loaded_revision != revision:
        raise ValueError("The resolved base model revision differs from the training manifest")
    resize_embeddings_for_tokenizer(base, tokenizer)
    adapted = PeftModel.from_pretrained(base, str(source), is_trainable=False)
    adapted.to(device)
    merged = adapted.merge_and_unload(safe_merge=True)
    merged.config.use_cache = True
    merged.generation_config.forced_bos_token_id = target_id
    merge_report = {
        "schema_version": 1,
        "adapter": str(source),
        "base_model": model_source,
        "base_model_sha256": base_digest,
        "base_revision_resolved": revision,
        "base_identity_verified": base_identity_verified,
        "warnings": []
        if base_identity_verified
        else [
            "The override could not be verified against the training base; "
            "the caller must confirm it has identical weights."
        ],
        "direction": direction,
        "context_token_id": context_id,
        "target_language_id": target_id,
        "training_identity_sha256": manifest.get("identity_sha256"),
        "smoke_test": manifest.get("smoke_test", False),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{target.name}.merge-", dir=target.parent) as temp:
        stage = Path(temp) / "model"
        stage.mkdir()
        merged.save_pretrained(str(stage), safe_serialization=True)
        tokenizer.save_pretrained(str(stage))
        (stage / "merge_manifest.json").write_text(
            json.dumps(merge_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if target.is_symlink() or (
            target.exists() and (not target.is_dir() or any(target.iterdir()))
        ):
            raise ValueError(f"Merge output changed or is not empty: {target}")
        if target.exists():
            target.rmdir()
        stage.rename(target)
    return merge_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", help="Override the base model recorded in the manifest")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args(argv)
    try:
        report = merge_adapter(
            args.adapter_dir,
            args.output_dir,
            base_model=args.base_model,
            local_files_only=args.local_files_only,
            device=args.device,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"Adapter merge failed: {exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
