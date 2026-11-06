"""Reconstruct source sentences and re-split translations at word boundaries."""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass

from ml.data.loaders import Cue


@dataclass(frozen=True, slots=True)
class CueSpan:
    cue_index: int
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class Sentence:
    text: str
    spans: tuple[CueSpan, ...]
    speaker: str | None

    @property
    def cue_ids(self) -> tuple[int, ...]:
        return tuple(span.cue_index for span in self.spans)


def _ends_sentence(text: str) -> bool:
    text = text.rstrip().rstrip("\"'”’»)]}").rstrip()
    if text.endswith(("...", "…", "-", "—", "–")):
        return False
    return text.endswith((".", "?", "!", "؟"))


def reconstruct_sentences(cues: list[Cue]) -> list[Sentence]:
    """Join adjacent incomplete cues and retain exact offsets in joined text.

    Original cue IDs and timestamps are never modified. An empty cue or an
    explicit change of speaker ends the current group.
    """
    sentences: list[Sentence] = []
    pieces: list[tuple[Cue, str]] = []
    current_speaker: str | None = None

    def flush() -> None:
        nonlocal current_speaker
        if not pieces:
            return
        text_parts: list[str] = []
        spans: list[CueSpan] = []
        offset = 0
        for cue, text in pieces:
            if text_parts:
                offset += 1
            start = offset
            text_parts.append(text)
            offset += len(text)
            spans.append(CueSpan(cue.index, start, offset))
        sentences.append(Sentence(" ".join(text_parts), tuple(spans), current_speaker))
        pieces.clear()
        current_speaker = None

    for cue in cues:
        text = " ".join(cue.text.split())
        if not text:
            flush()
            continue
        if cue.speaker and current_speaker and cue.speaker.casefold() != current_speaker.casefold():
            flush()
        if cue.speaker and not current_speaker:
            current_speaker = cue.speaker
        pieces.append((cue, text))
        if _ends_sentence(text):
            flush()
    flush()
    return sentences


_CONJUNCTIONS = frozenset({"and", "but", "or", "because", "so", "و", "بەڵام", "یان", "چونکە"})


def split_translation(sentence: Sentence, translated: str) -> list[str]:
    """Allocate target words proportionally to source spans, never mid-word.

    If there are fewer words than source cues, some output pieces are empty;
    this is exposed to QA rather than inventing or splitting a word.
    """
    if not sentence.spans:
        raise ValueError("A sentence must contain at least one cue span")
    words = re.findall(r"\S+", translated)
    count = len(sentence.spans)
    if count == 1:
        return [" ".join(words)]
    if not words:
        return [""] * count
    compact = " ".join(words)
    # positions[j] is the last character offset after j complete target words.
    positions = [0]
    for match in re.finditer(r"\S+", compact):
        positions.append(match.end())
    source_lengths = [max(1, span.end - span.start) for span in sentence.spans]
    total_source = sum(source_lengths)
    boundaries = [0]
    cumulative = 0
    total_words = len(words)
    for index in range(1, count):
        cumulative += source_lengths[index - 1]
        target = len(compact) * cumulative / total_source
        previous = boundaries[-1]
        if total_words >= count:
            lower = previous + 1
            upper = total_words - (count - index)
        else:
            lower, upper = previous, total_words
        nearest = bisect.bisect_left(positions, target, lo=lower, hi=upper + 1)
        candidates = {lower, upper}
        candidates.update(range(max(lower, nearest - 4), min(upper, nearest + 4) + 1))

        def rank(boundary: int, desired: float = target) -> tuple[float, float, int]:
            distance = abs(positions[boundary] - desired)
            previous_word = words[boundary - 1] if boundary else ""
            next_word = words[boundary].casefold() if boundary < total_words else ""
            preferred = previous_word.endswith(("،", "؛", ",", ";", ".", "؟", "?", "!"))
            preferred = preferred or next_word in _CONJUNCTIONS
            bonus = min(6.0, len(compact) / max(count, 1) * 0.12) if preferred else 0.0
            return distance - bonus, distance, boundary

        boundaries.append(min(candidates, key=rank))
    boundaries.append(total_words)
    return [
        " ".join(words[left:right])
        for left, right in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]
