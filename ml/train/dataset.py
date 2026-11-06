"""Film-scoped contextual examples and language-tagged NLLB tokenization."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

CONTEXT_TOKEN = "<ctx>"
PROVENANCE_WEIGHTS = {"human": 1.5, "unknown": 1.0, "machine": 0.5}
LANGUAGE_CODES = {"en-ckb": ("eng_Latn", "ckb_Arab"), "ckb-en": ("ckb_Arab", "eng_Latn")}
# Ellipses are continuations, not sentence endings. The heuristic cannot infer
# every abbreviation or utterance boundary; preserve film order and audit it.
_SENTENCE_END = re.compile(r"(?:(?<!\.)\.(?!\.)|[!?؟]+)[\"'”»\)\]]*(?=\s|$)")


@dataclass(frozen=True, slots=True)
class ContextExample:
    pair_id: str
    film_id: str
    source: str
    target: str
    context: tuple[str, ...]
    provenance: str
    sample_weight: float


def _protect_marker(text: str) -> str:
    # Do not let literal user content impersonate the delimiter special token.
    return text.replace(CONTEXT_TOKEN, "‹ctx›")


def _labelled(text: str, speakers: tuple[str, ...]) -> str:
    if not speakers:
        return _protect_marker(text)
    label = ", ".join(_protect_marker(speaker) for speaker in speakers)
    return f"[{label}] {_protect_marker(text)}"


def _completed_and_pending(text: str) -> tuple[list[str], str]:
    completed: list[str] = []
    last = 0
    for match in _SENTENCE_END.finditer(text):
        sentence = text[last : match.end()].strip()
        if sentence:
            completed.append(sentence)
        last = match.end()
    return completed, text[last:].strip()


def make_context_examples(
    rows: list[dict[str, Any]], *, direction: str, context_sentences: int = 3
) -> list[ContextExample]:
    """Form each target from its own pair and previous completed source sentences.

    Incomplete consecutive source cues are reconstructed into a sentence for
    subsequent context. The target is never prefixed with context. A speaker
    change flushes an unfinished utterance, preventing unrelated speakers from
    being joined. Context is reset at every film boundary.
    """
    if direction not in LANGUAGE_CODES or not 3 <= context_sentences <= 5:
        raise ValueError("Unsupported translation direction or context window")
    source_key, target_key = (
        ("english", "sorani") if direction == "en-ckb" else ("sorani", "english")
    )
    speaker_key = f"{source_key}_speakers"
    time_key = f"{source_key}_start_ms"
    ordered = sorted(rows, key=lambda row: (row["film_id"], row[time_key], row["pair_id"]))
    result: list[ContextExample] = []
    film: str | None = None
    completed: list[str] = []
    pending = ""
    pending_speakers: tuple[str, ...] = ()
    for row in ordered:
        row_film = row["film_id"]
        if row_film != film:
            film, completed, pending, pending_speakers = row_film, [], "", ()
        source = " ".join(row[source_key].split())
        target = " ".join(row[target_key].split())
        if not source or not target:
            raise ValueError(f"Pair {row['pair_id']} has empty source or target text")
        provenance = row["provenance"]
        if provenance not in PROVENANCE_WEIGHTS:
            raise ValueError(f"Pair {row['pair_id']} has invalid provenance {provenance!r}")
        speakers = tuple(dict.fromkeys(row.get(speaker_key) or []))
        if pending and pending_speakers and speakers and pending_speakers != speakers:
            completed.append(_labelled(pending, pending_speakers))
            pending, pending_speakers = "", ()
        context = (
            tuple(_protect_marker(" ".join(value.split())) for value in row["source_context"])
            if "source_context" in row
            else tuple(completed)
        )
        result.append(
            ContextExample(
                pair_id=row["pair_id"],
                film_id=row_film,
                source=_labelled(source, speakers),
                target=_protect_marker(target),
                context=context[-context_sentences:],
                provenance=provenance,
                sample_weight=PROVENANCE_WEIGHTS[provenance],
            )
        )
        if not pending_speakers:
            pending_speakers = speakers
        combined = f"{pending} {source}".strip()
        sentences, pending = _completed_and_pending(combined)
        completed.extend(_labelled(sentence, pending_speakers) for sentence in sentences)
        if not pending:
            pending_speakers = ()
    return result


def dataset_fingerprint(splits: dict[str, list[dict[str, Any]]]) -> str:
    """Hash the actual training-relevant records, without storing dialogue in a manifest."""
    digest = hashlib.sha256()
    keys = (
        "pair_id",
        "film_id",
        "english",
        "sorani",
        "english_start_ms",
        "sorani_start_ms",
        "english_speakers",
        "sorani_speakers",
        "provenance",
        "language_verified",
    )
    for split in ("train", "val", "test"):
        digest.update((split + "\n").encode())
        for row in sorted(
            splits.get(split, []), key=lambda item: (item["film_id"], item["pair_id"])
        ):
            values = [row.get(key) for key in keys]
            # Preserve existing fingerprints for legacy datasets without explicit context.
            if "source_context" in row:
                values.append(row["source_context"])
            digest.update(json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode())
            digest.update(b"\n")
    return digest.hexdigest()


def validate_splits(splits: dict[str, list[dict[str, Any]]], *, allow_unverified: bool) -> None:
    """Fail closed for missing labels, empty validation, or film leakage."""
    if not splits.get("train") or not splits.get("val"):
        raise ValueError("Training requires nonempty train and val splits")
    required = {
        "pair_id",
        "film_id",
        "english",
        "sorani",
        "english_start_ms",
        "sorani_start_ms",
        "provenance",
        "language_verified",
    }
    ids: set[str] = set()
    films: dict[str, set[str]] = {name: set() for name in ("train", "val", "test")}
    for name in films:
        for row in splits.get(name, []):
            missing = required - row.keys()
            if missing:
                raise ValueError(f"{name} row is missing fields: {', '.join(sorted(missing))}")
            for key in ("pair_id", "film_id", "english", "sorani"):
                if not isinstance(row[key], str) or not row[key].strip():
                    raise ValueError(f"{name} row has invalid {key}")
            for key in ("english_start_ms", "sorani_start_ms"):
                if isinstance(row[key], bool) or not isinstance(row[key], int) or row[key] < 0:
                    raise ValueError(f"{name} row has invalid {key}")
            if (
                not isinstance(row["provenance"], str)
                or row["provenance"] not in PROVENANCE_WEIGHTS
            ):
                raise ValueError(f"{name} row has invalid provenance")
            for key in ("english_speakers", "sorani_speakers"):
                speakers = row.get(key, [])
                if not isinstance(speakers, list) or not all(
                    isinstance(item, str) for item in speakers
                ):
                    raise ValueError(f"{name} row has invalid {key}")
            if "source_context" in row:
                context = row["source_context"]
                if (
                    not isinstance(context, list)
                    or len(context) > 5
                    or not all(isinstance(item, str) and item.strip() for item in context)
                ):
                    raise ValueError(f"{name} row has invalid source_context")
            if not isinstance(row["language_verified"], bool):
                raise ValueError(f"{name} row has invalid language_verified")
            if row["pair_id"] in ids:
                raise ValueError(f"Duplicate pair_id across dataset: {row['pair_id']}")
            ids.add(row["pair_id"])
            films[name].add(row["film_id"])
            if not row["language_verified"] and not allow_unverified:
                raise ValueError(
                    "Dataset contains language-unverified pairs. Verify with fastText or "
                    "explicitly pass --allow-unverified-data for an exploratory run."
                )
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if films[left] & films[right]:
            raise ValueError(f"Film leakage detected between {left} and {right}")


def token_ids(tokenizer: Any, direction: str) -> tuple[int, int, int, int, int]:
    if direction not in LANGUAGE_CODES:
        raise ValueError(f"Unsupported direction {direction!r}")
    src, tgt = LANGUAGE_CODES[direction]
    ids = tuple(tokenizer.convert_tokens_to_ids(token) for token in (src, tgt, CONTEXT_TOKEN))
    unknown = tokenizer.unk_token_id
    if any(value is None or value == unknown for value in ids):
        raise ValueError("Tokenizer must contain eng_Latn, ckb_Arab and the <ctx> special token")
    if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must define EOS and padding token IDs")
    return ids[0], ids[1], ids[2], tokenizer.eos_token_id, tokenizer.pad_token_id


def tokenize_example(
    example: ContextExample,
    tokenizer: Any,
    *,
    direction: str,
    max_source_length: int,
    max_target_length: int,
) -> dict[str, Any]:
    """Encode language tags and preserve the current cue ahead of old context.

    When context exists, reserve space for at least the newest context and a
    sentinel; then fill remaining room backwards from the newest sentence.
    Both context and source truncation are recorded for the training manifest.
    """
    src_id, tgt_id, ctx_id, eos_id, _ = token_ids(tokenizer, direction)
    budget = max_source_length - 2
    if budget < 2 or max_target_length < 3:
        raise ValueError("Token length limits are too small")
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)  # noqa: E731
    current = encode(example.source)
    target = encode(example.target)
    reserve = min(16, max(2, budget // 4)) if example.context else 0
    max_current = budget - reserve
    source_truncated = len(current) > max_current
    current = current[:max_current]
    available = budget - len(current)
    context: list[int] = []
    used = 0
    partial_context = False
    for sentence in reversed(example.context):
        if available < 2:
            break
        chunk = encode(sentence)
        if not chunk:
            continue
        take = min(len(chunk), available - 1)
        context = chunk[-take:] + [ctx_id] + context
        available -= take + 1
        used += 1
        if take < len(chunk):
            partial_context = True
            break
    inputs = [src_id, *context, *current, eos_id]
    labels = [tgt_id, *target[: max_target_length - 2], eos_id]
    return {
        "input_ids": inputs,
        "attention_mask": [1] * len(inputs),
        "labels": labels,
        "sample_weight": example.sample_weight,
        "pair_id": example.pair_id,
        "film_id": example.film_id,
        "context_sentences_available": len(example.context),
        "context_sentences_used": used,
        "context_truncated": partial_context or used < len(example.context),
        "source_truncated": source_truncated,
        "target_truncated": len(target) > max_target_length - 2,
    }
