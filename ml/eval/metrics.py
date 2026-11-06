"""Reproducible text, subtitle-speed, terminology and optional COMET metrics."""

from __future__ import annotations

import json
import math
import statistics
import unicodedata
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import regex
import sacrebleu
import sentencepiece
from sacrebleu.metrics import BLEU, CHRF

from ml.eval.records import EvaluationRecord, file_sha256


@dataclass(frozen=True, slots=True)
class GlossaryTerm:
    english: str
    sorani: str
    locked: bool = True


def load_glossary(path: Path) -> list[GlossaryTerm]:
    """Load a JSON array of locked or unlocked English/Sorani terms."""
    try:
        items = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read glossary JSON: {exc}") from exc
    if not isinstance(items, list):
        raise ValueError("Glossary must be a JSON array")
    terms: list[GlossaryTerm] = []
    seen: set[tuple[str, str]] = set()
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Glossary item {index} must be an object")
        english, sorani, locked = item.get("english"), item.get("sorani"), item.get("locked", True)
        if not isinstance(english, str) or not english.strip():
            raise ValueError(f"Glossary item {index} needs a nonempty english term")
        if not isinstance(sorani, str) or not sorani.strip():
            raise ValueError(f"Glossary item {index} needs a nonempty sorani term")
        if not isinstance(locked, bool):
            raise ValueError(f"Glossary item {index} locked must be boolean")
        key = (_normalized(english), _normalized(sorani))
        if key in seen:
            raise ValueError(f"Glossary item {index} is a duplicate")
        seen.add(key)
        terms.append(GlossaryTerm(english.strip(), sorani.strip(), locked))
    return terms


_BIDI_CONTROLS = frozenset(
    "\u061c\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"
)


def _normalized(text: str) -> str:
    text = "".join(char for char in text if char not in _BIDI_CONTROLS)
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def _contains_term(text: str, term: str) -> bool:
    # Match Unicode word boundaries rather than substrings inside another word.
    pattern = (
        r"(?<![\p{L}\p{N}\p{M}_])" + regex.escape(_normalized(term)) + r"(?![\p{L}\p{N}\p{M}_])"
    )
    return regex.search(pattern, _normalized(text)) is not None


def visible_characters(text: str) -> int:
    """Count visible grapheme clusters, ignoring line breaks and bidi controls."""
    clean = "".join(
        char
        for char in text
        if unicodedata.category(char) not in {"Cc", "Cf", "Zl", "Zp"} or char in "\u200c\u200d"
    )
    return sum(
        any(unicodedata.category(char) != "Cf" for char in cluster)
        for cluster in regex.findall(r"\X", clean)
    )


def cue_cps(text: str, duration_ms: int) -> float:
    if duration_ms <= 0:
        raise ValueError("CPS requires a positive duration")
    return visible_characters(text) / (duration_ms / 1000.0)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    point = (len(ordered) - 1) * fraction
    lower = math.floor(point)
    upper = math.ceil(point)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (point - lower)


def _spbleu(hypotheses: list[str], references: list[str], model_path: Path) -> dict[str, Any]:
    if not model_path.is_file():
        raise ValueError(f"SentencePiece model does not exist: {model_path}")
    try:
        tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(model_path))
        encode = lambda value: " ".join(tokenizer.encode(value, out_type=str))  # noqa: E731
        hyps = [encode(text) for text in hypotheses]
        refs = [encode(text) for text in references]
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"Could not load SentencePiece model: {exc}") from exc
    metric = BLEU(tokenize="none")
    score = metric.corpus_score(hyps, [refs])
    return {
        "score": float(score.score),
        "signature": str(metric.get_signature()),
        "sentencepiece_sha256": file_sha256(model_path),
        "sentencepiece_version": sentencepiece.__version__,
        "tokenization": "user-supplied SentencePiece model; SacreBLEU tokenize=none",
    }


def load_comet(checkpoint: Path, *, local_files_only: bool):
    if not checkpoint.is_file():
        raise ValueError(f"COMET checkpoint does not exist: {checkpoint}")
    try:
        from comet import load_from_checkpoint
    except ImportError as exc:
        raise ValueError(
            "COMET requires the optional eval dependencies: pip install '.[train,eval]'"
        ) from exc
    try:
        model = load_from_checkpoint(str(checkpoint), local_files_only=local_files_only)
    except Exception as exc:
        raise ValueError(f"Could not load the trusted COMET checkpoint: {exc}") from exc
    if getattr(model.hparams, "class_identifier", None) == "referenceless_regression_metric":
        raise ValueError("Use a reference-based COMET checkpoint, not a referenceless QE model")
    return model


def _comet(
    records: list[EvaluationRecord], hypotheses: list[str], model, *, batch_size: int, device: str
) -> dict[str, Any]:
    inputs = [
        {"src": record.source, "mt": hypothesis, "ref": record.reference}
        for record, hypothesis in zip(records, hypotheses, strict=True)
    ]
    output = model.predict(
        inputs,
        batch_size=batch_size,
        gpus=1 if device == "cuda" else 0,
        progress_bar=False,
        num_workers=0,
    )
    score = float(output.system_score)
    segment_scores = [float(item) for item in output.scores]
    if len(segment_scores) != len(records) or not all(
        math.isfinite(item) for item in [score, *segment_scores]
    ):
        raise ValueError("COMET returned invalid or incomplete scores")
    return {
        "score": score,
        "segment_count": len(segment_scores),
        "package_version": version("unbabel-comet"),
        "model_class": type(model).__name__,
        "requires_references": bool(model.requires_references()),
    }


def compute_metrics(
    records: list[EvaluationRecord],
    hypotheses: list[str],
    *,
    direction: str,
    glossary: list[GlossaryTerm] | None = None,
    sentencepiece_model: Path | None = None,
    comet_model=None,
    batch_size: int = 8,
    device: str = "cpu",
    target_cps: float = 17.0,
    hard_cps: float = 21.0,
) -> dict[str, Any]:
    """Score one hypothesis per record; unavailable optional metrics are null."""
    if not records or len(records) != len(hypotheses):
        raise ValueError("Metrics require a nonempty, aligned record and hypothesis list")
    if direction not in {"en-ckb", "ckb-en"}:
        raise ValueError("Unsupported translation direction")
    if any(not isinstance(hypothesis, str) for hypothesis in hypotheses):
        raise ValueError("Every hypothesis must be a string")
    if batch_size < 1 or device not in {"cpu", "cuda"}:
        raise ValueError("Invalid metric batch size or device")
    if not 0 < target_cps <= hard_cps:
        raise ValueError("CPS thresholds must be positive and target <= hard cap")
    references = [record.reference for record in records]
    chrf = CHRF(word_order=2)
    chrf_score = chrf.corpus_score(hypotheses, [references])
    cps_values = [
        cue_cps(hyp, record.duration_ms) for record, hyp in zip(records, hypotheses, strict=True)
    ]
    target_count = sum(value > target_cps for value in cps_values)
    hard_count = sum(value > hard_cps for value in cps_values)
    ratios = [
        visible_characters(hyp) / max(1, visible_characters(ref))
        for hyp, ref in zip(hypotheses, references, strict=True)
    ]
    applicable = matched = 0
    for record, hypothesis in zip(records, hypotheses, strict=True):
        for term in glossary or []:
            if not term.locked:
                continue
            source_term, target_term = (
                (term.english, term.sorani)
                if direction == "en-ckb"
                else (term.sorani, term.english)
            )
            if _contains_term(record.source, source_term):
                applicable += 1
                matched += _contains_term(hypothesis, target_term)
    return {
        "count": len(records),
        "chrfpp": float(chrf_score.score),
        "chrf_signature": str(chrf.get_signature()),
        "sacrebleu_version": sacrebleu.__version__,
        "spbleu": _spbleu(hypotheses, references, sentencepiece_model)
        if sentencepiece_model
        else None,
        "comet": _comet(records, hypotheses, comet_model, batch_size=batch_size, device=device)
        if comet_model is not None
        else None,
        "cps": {
            "target": target_cps,
            "hard_cap": hard_cps,
            "above_target_count": target_count,
            "above_target_percent": 100.0 * target_count / len(records),
            "above_hard_cap_count": hard_count,
            "above_hard_cap_percent": 100.0 * hard_count / len(records),
            "basis": "original source aligned-group duration; visible grapheme clusters",
        },
        "glossary": {
            "provided": glossary is not None,
            "applicable_locked_terms": applicable,
            "matched_locked_terms": matched,
            "adherence_percent": 100.0 * matched / applicable if applicable else None,
        },
        "length_ratio_to_reference": {
            "min": min(ratios),
            "mean": statistics.fmean(ratios),
            "median": statistics.median(ratios),
            "p90": _percentile(ratios, 0.9),
            "max": max(ratios),
        },
    }
