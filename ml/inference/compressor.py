"""Conservative, reviewable reading-speed compression rules."""

from __future__ import annotations

import math
from dataclasses import dataclass

import regex

from ml.inference.glossary import contains_term
from ml.inference.text import cps, visible_characters


@dataclass(frozen=True, slots=True)
class CompressionResult:
    text: str
    applied_rules: tuple[str, ...]
    still_above_target: bool


_RULES = {
    "en-ckb": (
        ("remove_hesitation", regex.compile(r"^(?:ئەممم+|ئێێ+)[،,]?\s+", regex.IGNORECASE), ""),
        ("concise_cause", regex.compile(r"(?<!\p{L})لەبەر ئەوەی(?!\p{L})"), "چونکە"),
        ("remove_repeated_pronoun", regex.compile(r"^(من)\s+\1(?=\s)", regex.IGNORECASE), r"\1"),
    ),
    "ckb-en": (
        ("remove_hesitation", regex.compile(r"^(?:um+|uh+|erm+)[,]?\s+", regex.IGNORECASE), ""),
        ("concise_purpose", regex.compile(r"\bin order to\b", regex.IGNORECASE), "to"),
        ("concise_cause", regex.compile(r"\bdue to the fact that\b", regex.IGNORECASE), "because"),
        ("remove_repeated_pronoun", regex.compile(r"^(I)\s+\1(?=\s)", regex.IGNORECASE), r"\1"),
    ),
}


def compress(
    text: str,
    duration_ms: int,
    *,
    direction: str,
    target_cps: float = 17.0,
    protected_terms: tuple[str, ...] = (),
) -> CompressionResult:
    """Shorten only over-speed text without dropping words to force a hard cap.

    Rules are intentionally limited and are not proof of semantic equivalence;
    every edit is recorded so a human can review it. Locked terms already
    present in the cue may not be removed by a rule.
    """
    if (
        not isinstance(direction, str)
        or direction not in _RULES
        or isinstance(target_cps, bool)
        or not isinstance(target_cps, (int, float))
        or not math.isfinite(target_cps)
        or target_cps <= 0
    ):
        raise ValueError("Unsupported compression direction or target CPS")
    value = " ".join(text.split())
    applied: list[str] = []
    for name, pattern, replacement in _RULES[direction]:
        if cps(value, duration_ms) <= target_cps:
            break
        candidate = pattern.sub(replacement, value, count=1).strip()
        if not candidate or visible_characters(candidate) >= visible_characters(value):
            continue
        if any(
            contains_term(value, term) and not contains_term(candidate, term)
            for term in protected_terms
        ):
            continue
        value = candidate
        applied.append(name)
    return CompressionResult(value, tuple(applied), cps(value, duration_ms) > target_cps)
