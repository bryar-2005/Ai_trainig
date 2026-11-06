"""Batched NLLB/M2M100 translation with optional Phase 2 LoRA adapter."""

from __future__ import annotations

import hashlib
import json
import math
import re
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from ml.inference.base import BaseTranslator, TranslationRequest, TranslationResult
from ml.inference.model_source import resolve_model_source
from ml.train.dataset import ContextExample, token_ids, tokenize_example
from ml.train.model_utils import (
    add_context_token,
    local_model_digest,
    resize_embeddings_for_tokenizer,
)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read model manifest: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("Model manifest must be a JSON object")
    return value


def _protect(text: str) -> str:
    return " ".join(text.split()).replace("<ctx>", "‹ctx›")


class NllbTranslator(BaseTranslator):
    """Use real Transformers generation and selected-token transition scores.

    The adapter, when supplied, must be a completed Phase 2 adapter and must
    match the recorded local base hash or immutable Hub revision. Base-only
    checkpoints without <ctx> receive a new untrained marker row.
    """

    def __init__(
        self,
        *,
        direction: str,
        model_version_id: str,
        model: str | None = None,
        adapter_dir: str | Path | None = None,
        model_revision: str | None = None,
        local_files_only: bool = False,
        allow_smoke_model: bool = False,
        device: str = "cpu",
        batch_size: int = 4,
        max_source_length: int = 256,
        max_target_length: int = 256,
        num_beams: int = 4,
        context_window: int = 3,
    ) -> None:
        if direction not in {"en-ckb", "ckb-en"}:
            raise ValueError("Unsupported translation direction")
        if not isinstance(model_version_id, str) or not model_version_id.strip():
            raise ValueError("model_version_id must be nonempty")
        if device not in {"cpu", "cuda"} or batch_size < 1:
            raise ValueError("Invalid translation device or batch size")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        if (
            min(max_source_length, max_target_length) < 4
            or num_beams < 1
            or not 3 <= context_window <= 5
        ):
            raise ValueError("Invalid model length limits, beam count or context window")
        if model_revision is not None and (
            not isinstance(model_revision, str) or not re.fullmatch(r"[0-9a-f]{40}", model_revision)
        ):
            raise ValueError("model_revision must be a 40-character immutable Hub commit")
        if adapter_dir is not None and model_revision is not None:
            raise ValueError("Adapters use the immutable base revision recorded in their manifest")
        self.direction, self.model_version_id = direction, model_version_id.strip()
        self.device, self.batch_size = device, batch_size
        self.context_window = context_window
        self.warnings: list[str] = []
        manifest: dict[str, Any] = {}
        merged: dict[str, Any] = {}
        adapter: Path | None = None
        revision: str | None = None
        if adapter_dir is not None:
            adapter = Path(adapter_dir).expanduser().resolve()
            if (
                not (adapter / "adapter_config.json").is_file()
                or not (adapter / "adapter_model.safetensors").is_file()
            ):
                raise ValueError("Adapter directory must contain Phase 2 PEFT files")
            manifest = _read_manifest(adapter / "training_manifest.json")
            settings = manifest.get("config")
            if manifest.get("status") != "completed" or not isinstance(settings, dict):
                raise ValueError("Adapter training manifest is incomplete")
            if settings.get("direction") != direction:
                raise ValueError("Adapter direction does not match the request")
            if manifest.get("smoke_test") and not allow_smoke_model:
                raise ValueError("Random smoke models require --allow-smoke-model")
            recorded = manifest.get("actual_base_model")
            source = model or recorded
            if not isinstance(source, str) or not source:
                raise ValueError("Adapter manifest does not identify the base model")
            if Path(source).expanduser().is_dir():
                source = str(Path(source).expanduser().resolve())
            expected = manifest.get("base_model_sha256")
            digest = local_model_digest(source)
            if expected is not None:
                if digest != expected:
                    raise ValueError("Local base model does not match the adapter hash")
            else:
                revision = manifest.get("base_revision_resolved")
                if not revision or source != recorded:
                    raise ValueError("The adapter's Hub base revision cannot be verified")
            training = settings.get("training")
            if not isinstance(training, dict):
                raise ValueError("Adapter manifest is missing training settings")
            try:
                max_source_length = int(settings["max_source_length"])
                max_target_length = int(settings["max_target_length"])
                self.context_window = int(settings["context_sentences"])
                num_beams = int(training["generation_num_beams"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Adapter manifest contains invalid generation settings") from exc
            if (
                not 3 <= self.context_window <= 5
                or min(max_source_length, max_target_length) < 4
                or num_beams < 1
            ):
                raise ValueError("Adapter manifest contains unsupported generation settings")
            tokenizer_source = str(adapter)
            if manifest.get("smoke_test"):
                self.warnings.append("Random smoke model: output is not a usable translation.")
        else:
            if not isinstance(model, str) or not model:
                raise ValueError(
                    "A base or merged model path is required when no adapter is supplied"
                )
            source = (
                str(Path(model).expanduser().resolve())
                if Path(model).expanduser().is_dir()
                else model
            )
            tokenizer_source = source
            digest = local_model_digest(source)
            if digest is not None and model_revision is not None:
                raise ValueError(
                    "model_revision applies to Hub models, not local model directories"
                )
            revision = model_revision
            merged_path = Path(source) / "merge_manifest.json"
            if merged_path.is_file():
                merged = _read_manifest(merged_path)
                if merged.get("direction") != direction:
                    raise ValueError("Merged model direction does not match the request")
                if merged.get("smoke_test") and not allow_smoke_model:
                    raise ValueError("Random smoke models require --allow-smoke-model")
                if merged.get("smoke_test"):
                    self.warnings.append("Random smoke model: output is not a usable translation.")
                if merged.get("base_identity_verified") is False:
                    self.warnings.append(
                        "The merged model's original base identity was not verified."
                    )
        self.max_source_length, self.max_target_length, self.num_beams = (
            max_source_length,
            max_target_length,
            num_beams,
        )
        load_source, cached_revision = resolve_model_source(
            source, revision=revision, local_files_only=local_files_only
        )
        if (Path(load_source) / "smoke_model.json").is_file():
            if not allow_smoke_model:
                raise ValueError("Random smoke models require --allow-smoke-model")
            warning = "Random smoke model: output is not a usable translation."
            if warning not in self.warnings:
                self.warnings.append(warning)
        model_kwargs: dict[str, Any] = {}
        if revision and load_source == source and digest is None:
            # Ensure PEFT's optional adapter-config check uses the same Hub revision.
            model_kwargs["adapter_kwargs"] = {"revision": revision}
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            load_source,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=False,
            **model_kwargs,
        )
        actual_revision = getattr(self.model.config, "_commit_hash", None) or cached_revision
        if revision and actual_revision and revision != actual_revision:
            raise ValueError("Loaded base revision does not match the requested model revision")
        position_limit = getattr(self.model.config, "max_position_embeddings", None)
        if (
            isinstance(position_limit, int)
            and max(max_source_length, max_target_length) > position_limit
        ):
            raise ValueError("Requested token length exceeds the model's position limit")
        tokenizer_revision = None if adapter else (revision or actual_revision)
        if not adapter:
            tokenizer_source = load_source
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            revision=tokenizer_revision,
            local_files_only=local_files_only,
            trust_remote_code=False,
        )
        context_was_present = "<ctx>" in self.tokenizer.get_vocab()
        add_context_token(self.tokenizer)
        _, self.target_id, context_id, _, _ = token_ids(self.tokenizer, direction)
        if adapter and (
            context_id != manifest.get("context_token_id")
            or self.target_id != manifest.get("target_language_id")
        ):
            raise ValueError("Adapter tokenizer IDs do not match its training manifest")
        if merged and (
            context_id != merged.get("context_token_id")
            or self.target_id != merged.get("target_language_id")
        ):
            raise ValueError("Merged tokenizer IDs do not match its merge manifest")
        untrained_marker = (
            not context_was_present
            or context_id >= self.model.get_input_embeddings().num_embeddings
        )
        # Resizing happens on CPU; a fixed seed makes an untrained base marker
        # reproducible without mutating the host application's RNG state.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            resize_embeddings_for_tokenizer(self.model, self.tokenizer)
        if adapter:
            self.model = PeftModel.from_pretrained(self.model, str(adapter), is_trainable=False)
        elif untrained_marker:
            self.warnings.append("The base model's new <ctx> marker is untrained.")
        self.model.config.use_cache = True
        self.model.generation_config.forced_bos_token_id = self.target_id
        self.model.to(device).eval()
        if digest is None and not (actual_revision or revision):
            self.warnings.append("The base model revision could not be established automatically.")
        self.model_info = {
            "source": source,
            "resolved_source": load_source,
            "base_sha256": digest,
            "resolved_revision": actual_revision or revision,
            "adapter": str(adapter) if adapter else None,
            "adapter_sha256": _file_hash(adapter / "adapter_model.safetensors")
            if adapter
            else None,
            "model_version_id": self.model_version_id,
            "context_window": self.context_window,
            "context_token_id": context_id,
            "target_language_id": self.target_id,
            "max_source_length": max_source_length,
            "max_target_length": max_target_length,
            "num_beams": num_beams,
            "device": device,
            "libraries": {
                "torch": str(torch.__version__),
                "transformers": version("transformers"),
                "peft": version("peft"),
            },
            "warnings": list(self.warnings),
        }

    def _encoded(self, request: TranslationRequest, number: int) -> dict[str, Any]:
        if not isinstance(request.text, str) or not request.text.strip():
            raise ValueError("Translation requests require nonempty source text")
        speaker = f"[{_protect(request.speaker)}] " if request.speaker else ""
        hints = [
            f"[Glossary: {_protect(source)} = {_protect(target)}]"
            for source, target in request.glossary_hints
        ]
        example = ContextExample(
            pair_id=str(number),
            film_id="inference",
            source=speaker + _protect(request.text),
            target=".",
            context=tuple(_protect(text) for text in request.additional_context)
            + tuple(_protect(text) for text in request.context[-self.context_window :])
            + tuple(hints),
            provenance="unknown",
            sample_weight=1.0,
        )
        return tokenize_example(
            example,
            self.tokenizer,
            direction=self.direction,
            max_source_length=self.max_source_length,
            max_target_length=self.max_target_length,
        )

    def _generate(
        self, requests: list[TranslationRequest], encoded: list[dict[str, Any]]
    ) -> list[TranslationResult]:
        features = [{key: row[key] for key in ("input_ids", "attention_mask")} for row in encoded]
        inputs = self.tokenizer.pad(features, padding=True, return_tensors="pt").to(self.device)
        kwargs: dict[str, Any] = {
            "forced_bos_token_id": self.target_id,
            "max_length": self.max_target_length,
            "num_beams": self.num_beams,
        }
        if requests[0].force_terms:
            if len(requests) != 1:
                raise ValueError("Constrained generation requires an individual request")
            constraints = []
            for term in dict.fromkeys(requests[0].force_terms):
                ids = self.tokenizer.encode(term, add_special_tokens=False)
                if not ids or self.tokenizer.unk_token_id in ids:
                    raise ValueError("A locked term cannot be represented by this tokenizer")
                constraints.append(ids)
            if len(constraints) > 16 or sum(map(len, constraints)) + 2 > self.max_target_length:
                raise ValueError("Locked terms exceed the constrained generation budget")
            kwargs["force_words_ids"] = constraints
            kwargs["num_beams"] = max(2, self.num_beams)
        with torch.inference_mode():
            sequences = self.model.generate(**inputs, **kwargs)
            # Re-score only the selected sequence, one decoder step at a time.
            # Keeping every beam's vocabulary-sized logits for every step can
            # exceed several GB with NLLB's 256k-token vocabulary.
            encoder = self.model.get_encoder()(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                return_dict=True,
            )
            values: list[list[float]] = [[] for _ in requests]
            invalid_scores = [False for _ in requests]
            alive = [True for _ in requests]
            special_ids = set(self.tokenizer.all_special_ids)
            past = None
            for position in range(sequences.shape[1] - 1):
                result = self.model(
                    encoder_outputs=encoder,
                    attention_mask=inputs["attention_mask"],
                    decoder_input_ids=sequences[:, position : position + 1],
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
                past = result.past_key_values
                tokens = sequences[:, position + 1]
                scores = (
                    torch.log_softmax(result.logits[:, -1, :].float(), dim=-1)
                    .gather(1, tokens.unsqueeze(1))
                    .squeeze(1)
                )
                for index, (token, score) in enumerate(
                    zip(tokens.tolist(), scores.tolist(), strict=True)
                ):
                    if token == self.tokenizer.eos_token_id:
                        alive[index] = False
                    elif alive[index] and token not in special_ids:
                        if math.isfinite(float(score)):
                            values[index].append(float(score))
                        else:
                            invalid_scores[index] = True
        decoded = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
        results = []
        for index, (text, scores, row) in enumerate(zip(decoded, values, encoded, strict=True)):
            results.append(
                TranslationResult(
                    text.strip(),
                    tuple(scores),
                    {
                        "source_truncated": row["source_truncated"],
                        "context_truncated": row["context_truncated"],
                        "context_sentences_used": min(
                            len(requests[index].context[-self.context_window :]),
                            max(
                                0,
                                row["context_sentences_used"] - len(requests[index].glossary_hints),
                            ),
                        ),
                        "additional_context_used": min(
                            len(requests[index].additional_context),
                            max(
                                0,
                                row["context_sentences_used"]
                                - len(requests[index].glossary_hints)
                                - len(requests[index].context[-self.context_window :]),
                            ),
                        ),
                        "glossary_hints_used": min(
                            len(requests[index].glossary_hints), row["context_sentences_used"]
                        ),
                        "invalid_scores": invalid_scores[index],
                        "constrained": bool(requests[index].force_terms),
                    },
                )
            )
        return results

    def translate_batch(self, requests: list[TranslationRequest]) -> list[TranslationResult]:
        if not requests:
            return []
        encoded = [self._encoded(request, index) for index, request in enumerate(requests)]
        result: list[TranslationResult] = []
        index = 0
        while index < len(requests):
            if requests[index].force_terms:
                stop = index + 1
            else:
                stop = index
                while (
                    stop < min(len(requests), index + self.batch_size)
                    and not requests[stop].force_terms
                ):
                    stop += 1
            result.extend(self._generate(requests[index:stop], encoded[index:stop]))
            index = stop
        return result
