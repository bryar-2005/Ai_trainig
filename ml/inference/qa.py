"""Cue and project QA with explicitly uncalibrated confidence estimates."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import regex

from ml.data.normalize import normalization_warnings
from ml.inference.text import cps, visible_characters


@dataclass(frozen=True, slots=True)
class CueQuality:
    cps: float
    line_lengths: tuple[int, ...]
    length_ratio: float | None
    mean_token_logprob: float | None
    confidence_score: float
    warnings: tuple[str, ...]
    flagged: bool


def score_cue(
    *,
    source: str,
    target: str,
    duration_ms: int,
    direction: str,
    token_logprobs: tuple[float, ...] = (),
    missing_terms: tuple[str, ...] = (),
    target_cps: float = 17.0,
    hard_cps: float = 21.0,
    max_chars: int = 42,
    max_lines: int = 2,
    timing_overlap: bool = False,
    extra_warnings: tuple[str, ...] = (),
) -> CueQuality:
    if (
        not isinstance(direction, str)
        or direction not in {"en-ckb", "ckb-en"}
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in (target_cps, hard_cps)
        )
        or not 0 < target_cps <= hard_cps
    ):
        raise ValueError("Invalid QA direction or CPS thresholds")
    if max_chars < 1 or max_lines < 1:
        raise ValueError("Invalid line limits")
    speed = cps(target, duration_ms)
    lines = target.splitlines() or [""]
    lengths = tuple(visible_characters(line) for line in lines)
    source_length, target_length = visible_characters(source), visible_characters(target)
    ratio = target_length / source_length if source_length else None
    valid_probs = [float(value) for value in token_logprobs if math.isfinite(value) and value <= 0]
    average = sum(valid_probs) / len(valid_probs) if valid_probs else None
    warnings: list[str] = []

    def flag(value: str) -> None:
        if value not in warnings:
            warnings.append(value)

    for item in extra_warnings:
        flag(item)
    if source.strip() and not target.strip():
        flag("missing_translation")
    if speed > target_cps:
        flag("cps_above_target")
    if speed > hard_cps:
        flag("cps_above_hard_cap")
    if len(lines) > max_lines:
        flag("too_many_lines")
    if any(length > max_chars for length in lengths):
        flag("line_too_long")
    if ratio is not None and (ratio < 0.4 or ratio > 2.5):
        flag("length_ratio_anomaly")
    if missing_terms:
        flag("glossary_mismatch")
    if direction == "en-ckb":
        if regex.search(r"\p{Latin}", target):
            flag("untranslated_latin")
        for warning in normalization_warnings(target):
            flag(warning)
    elif regex.search(r"\p{Script=Arabic}", target):
        flag("untranslated_arabic_script")
    if timing_overlap:
        flag("original_timing_overlap")
    if source.strip() and average is None:
        flag("model_logprob_unavailable")

    # This combines signals monotonically; it is not a calibrated probability
    # of a correct translation. Sentence-level log probabilities are shared
    # by cues reconstructed from that sentence.
    confidence = (
        100.0
        if not source.strip()
        else (100.0 * math.exp(average / 3) if average is not None else 50.0)
    )
    penalties = {
        "missing_translation": 45,
        "cps_above_target": 5,
        "cps_above_hard_cap": 15,
        "too_many_lines": 10,
        "line_too_long": 10,
        "length_ratio_anomaly": 15,
        "glossary_mismatch": 20,
        "untranslated_latin": 10,
        "untranslated_arabic_script": 10,
        "original_timing_overlap": 5,
        "original_timing_out_of_order": 5,
        "ambiguous_arabic_heh": 3,
        "source_truncated": 15,
        "context_truncated": 5,
        "model_logprob_unavailable": 10,
        "invalid_model_scores": 10,
        "compression_needs_review": 5,
        "sfx_untranslated": 5,
    }
    confidence = max(0.0, min(100.0, confidence - sum(penalties.get(item, 0) for item in warnings)))
    return CueQuality(
        cps=speed,
        line_lengths=lengths,
        length_ratio=ratio,
        mean_token_logprob=average,
        confidence_score=round(confidence, 2),
        warnings=tuple(warnings),
        flagged=bool(warnings) or confidence < 70,
    )


def summarize_qa(cues: list[dict]) -> dict:
    counts: Counter[str] = Counter()
    for cue in cues:
        counts.update(cue["qa_warnings"])
    return {
        "cue_count": len(cues),
        "flagged_count": sum(bool(cue["flagged"]) for cue in cues),
        "average_confidence": (
            sum(cue["confidence_score"] for cue in cues) / len(cues) if cues else None
        ),
        "warning_counts": dict(sorted(counts.items())),
        "confidence_note": (
            "Uncalibrated heuristic; sentence-level model log-probs are not "
            "correctness probabilities."
        ),
    }
