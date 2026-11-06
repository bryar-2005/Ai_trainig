from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from ml.train.dataset import (
    CONTEXT_TOKEN,
    ContextExample,
    dataset_fingerprint,
    make_context_examples,
    token_ids,
    tokenize_example,
    validate_splits,
)
from ml.train.model_utils import create_smoke_base, local_model_digest


def row(film: str, number: int, english: str, *, speaker: str = "JOHN", verified: bool = True):
    return {
        "pair_id": f"{film}-{number}",
        "film_id": film,
        "english": english,
        "sorani": f"ئەم دەقە بۆ تاقیکردنەوەی ژمارە {number}.",
        "english_start_ms": number * 1000,
        "sorani_start_ms": number * 1000,
        "english_speakers": [speaker] if speaker else [],
        "sorani_speakers": [],
        "provenance": ["human", "unknown", "machine"][number % 3],
        "language_verified": verified,
    }


def test_context_reconstructs_previous_sentences_keeps_speakers_and_resets_by_film():
    rows = [
        row("a", 0, "We need to go"),
        row("a", 1, "home now. Be careful!"),
        row("a", 2, "Which way?"),
        row("a", 3, "Over there.", speaker="MARY"),
        row("b", 0, "A different film."),
    ]
    examples = make_context_examples(list(reversed(rows)), direction="en-ckb")
    assert examples[0].context == ()
    assert examples[1].context == ()
    assert examples[2].context == ("[JOHN] We need to go home now.", "[JOHN] Be careful!")
    assert examples[3].context[-1] == "[JOHN] Which way?"
    assert examples[3].source == "[MARY] Over there."
    assert examples[4].context == ()
    assert [example.sample_weight for example in examples[:3]] == [1.5, 1.0, 0.5]
    assert CONTEXT_TOKEN not in examples[3].target


def test_speaker_change_flushes_incomplete_utterance_and_ellipsis_continues():
    rows = [
        row("a", 0, "Wait..."),
        row("a", 1, "I need to", speaker="JOHN"),
        row("a", 2, "Stop!", speaker="MARY"),
    ]
    examples = make_context_examples(rows, direction="en-ckb")
    assert examples[1].context == ()
    assert examples[2].context == ("[JOHN] Wait... I need to",)


def test_reverse_direction_uses_sorani_as_source():
    rows = [row("a", 0, "First."), row("a", 1, "Second.")]
    examples = make_context_examples(rows, direction="ckb-en")
    assert examples[0].source == rows[0]["sorani"]
    assert examples[0].target == "First."
    assert examples[1].context == (rows[0]["sorani"],)


def test_train_split_validation_and_fingerprint():
    splits = {"train": [row("a", 0, "Hi.")], "val": [row("b", 0, "Hello.")], "test": []}
    validate_splits(splits, allow_unverified=False)
    fingerprint = dataset_fingerprint(splits)
    splits["train"][0]["english"] = "Changed."
    assert dataset_fingerprint(splits) != fingerprint
    splits["val"][0]["film_id"] = "a"
    with pytest.raises(ValueError, match="Film leakage"):
        validate_splits(splits, allow_unverified=False)
    splits["val"][0]["film_id"] = "b"
    splits["train"][0]["language_verified"] = False
    with pytest.raises(ValueError, match="unverified"):
        validate_splits(splits, allow_unverified=False)
    validate_splits(splits, allow_unverified=True)


def test_explicit_source_context_is_validated_protected_and_fingerprinted():
    item = row("a", 2, "Current sentence.")
    item["source_context"] = ["Intervening source.", "Literal <ctx> text."]
    splits = {"train": [item], "val": [row("b", 0, "Validation.")], "test": []}
    validate_splits(splits, allow_unverified=False)
    example = make_context_examples([item], direction="en-ckb")[0]
    assert example.context == ("Intervening source.", "Literal ‹ctx› text.")
    fingerprint = dataset_fingerprint(splits)
    item["source_context"][0] = "Changed source."
    assert dataset_fingerprint(splits) != fingerprint
    item["source_context"] = [""]
    with pytest.raises(ValueError, match="source_context"):
        validate_splits(splits, allow_unverified=False)


def test_tokenization_keeps_newest_context_and_target_labels_separate(tmp_path: Path):
    example = ContextExample(
        pair_id="1",
        film_id="film",
        source="[JOHN] " + "current " * 50,
        target="سڵاو هاوڕێ",
        context=("old " * 70, "recent " * 70),
        provenance="human",
        sample_weight=1.5,
    )
    tokenizer, _ = create_smoke_base([example], tmp_path / "base", seed=42)
    encoded = tokenize_example(
        example, tokenizer, direction="en-ckb", max_source_length=24, max_target_length=16
    )
    src_id, tgt_id, ctx_id, eos_id, _ = token_ids(tokenizer, "en-ckb")
    assert len(encoded["input_ids"]) <= 24
    assert encoded["input_ids"][0] == src_id
    assert encoded["input_ids"][-1] == eos_id
    assert ctx_id in encoded["input_ids"]
    assert ctx_id not in encoded["labels"]
    assert encoded["labels"][0] == tgt_id
    assert encoded["context_sentences_used"] == 1
    assert encoded["context_truncated"] is True
    assert encoded["source_truncated"] is True
    assert tokenizer.encode(CONTEXT_TOKEN, add_special_tokens=False) == [ctx_id]
    only_one = tokenize_example(
        replace(example, context=("recent " * 70,)),
        tokenizer,
        direction="en-ckb",
        max_source_length=24,
        max_target_length=16,
    )
    assert only_one["context_sentences_used"] == only_one["context_sentences_available"] == 1
    assert only_one["context_truncated"] is True


def test_user_text_cannot_inject_the_context_marker():
    example = make_context_examples([row("a", 0, "I literally typed <ctx>.")], direction="en-ckb")[
        0
    ]
    assert CONTEXT_TOKEN not in example.source
    assert "‹ctx›" in example.source


def test_local_base_fingerprint_tracks_model_files_without_recording_content(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text('{"hidden": 4}', encoding="utf-8")
    (base / "model.safetensors").write_bytes(b"original synthetic weights")
    original = local_model_digest(base)
    assert len(original) == 64
    (base / "unrelated.txt").write_text("ignored", encoding="utf-8")
    assert local_model_digest(base) == original
    (base / "model.safetensors").write_bytes(b"changed synthetic weights")
    assert local_model_digest(base) != original
