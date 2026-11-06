"""Unicode-aware visible length, safe directionality, and subtitle wrapping."""

from __future__ import annotations

import math
import unicodedata

import regex

from ml.data.normalize import normalize_english, normalize_sorani

LRI = "\u2066"
PDI = "\u2069"
_BIDI = frozenset("\u061c\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069")
_LTR_RUN = regex.compile(
    r"\p{Latin}[\p{Latin}\p{M}0-9]*(?:[.'’_-][\p{Latin}\p{M}0-9]+)*"
    r"|[0-9\u0660-\u0669\u06f0-\u06f9]+(?:[.:/,-][0-9\u0660-\u0669\u06f0-\u06f9]+)*"
)


def visible_characters(text: str) -> int:
    """Count graphemes including spaces, excluding line breaks and controls."""
    clean = "".join(
        character
        for character in text
        if unicodedata.category(character) not in {"Cc", "Cf", "Zl", "Zp"}
        or character in "\u200c\u200d"
    )
    return sum(
        any(unicodedata.category(character) != "Cf" for character in cluster)
        for cluster in regex.findall(r"\X", clean)
    )


def cps(text: str, duration_ms: int) -> float:
    if (
        isinstance(duration_ms, bool)
        or not isinstance(duration_ms, (int, float))
        or not math.isfinite(duration_ms)
        or duration_ms <= 0
    ):
        raise ValueError("Cue duration must be finite and positive")
    return 1000.0 * visible_characters(text) / duration_ms


def normalize_target(text: str, direction: str) -> str:
    if direction == "en-ckb":
        return normalize_sorani(text)
    if direction == "ckb-en":
        return normalize_english(text)
    raise ValueError("Unsupported translation direction")


def isolate_ltr(text: str) -> str:
    """Isolate embedded Latin/numeric runs; stripping existing controls is idempotent."""
    safe = "".join(character for character in text if character not in _BIDI)
    return _LTR_RUN.sub(lambda match: f"{LRI}{match.group()}{PDI}", safe)


def wrap_lines(text: str, *, max_chars: int = 42, max_lines: int = 2) -> str:
    """Choose at most one whitespace break, favoring balance and syntax.

    Never split a word or discard characters to meet the limit. If no valid
    break can meet it, the best available lines are returned for QA.
    """
    if (
        isinstance(max_chars, bool)
        or not isinstance(max_chars, int)
        or max_chars < 1
        or isinstance(max_lines, bool)
        or not isinstance(max_lines, int)
        or max_lines not in {1, 2}
    ):
        raise ValueError("max_chars must be positive and max_lines must be 1 or 2")
    text = " ".join(text.split())
    if not text or max_lines == 1 or visible_characters(text) <= max_chars:
        return text
    words = text.split(" ")
    if len(words) < 2:
        return text
    lengths = [visible_characters(word) for word in words]
    total = sum(lengths) + len(words) - 1
    left_length = 0
    position = 0
    candidates: list[tuple[int, int, int]] = []
    for index, word in enumerate(words[:-1]):
        left_length += lengths[index] + (1 if index else 0)
        position += len(word) + (1 if index else 0)
        right_length = total - left_length - 1
        overflow = max(0, left_length - max_chars) + max(0, right_length - max_chars)
        boundary = word.endswith(("،", "؛", ",", ";", ":", ".", "؟", "?", "!"))
        next_word = words[index + 1].casefold()
        syntactic = boundary or next_word in {"and", "but", "or", "because", "و", "بەڵام", "یان"}
        candidates.append(
            (overflow, abs(left_length - right_length) - (2 if syntactic else 0), position)
        )
    position = min(candidates)[2]
    return text[:position] + "\n" + text[position + 1 :]
