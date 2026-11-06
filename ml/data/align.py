"""Order-preserving temporal alignment of original subtitle cues.

Candidate groups are 1:1, 1:n, or n:1. Both outer boundaries must be within
``tolerance_ms``, and IoU is computed on the *original*, unshifted spans. A
sparse weighted-chain dynamic program chooses non-overlapping matches globally;
it does not greedily consume a cue that would make a better later match fail.
"""

from __future__ import annotations

import bisect
import heapq
from dataclasses import dataclass

from ml.data.loaders import Cue


@dataclass(frozen=True, slots=True)
class AlignedPair:
    english_cues: tuple[Cue, ...]
    sorani_cues: tuple[Cue, ...]
    temporal_iou: float


def temporal_iou(start_a: int, end_a: int, start_b: int, end_b: int) -> float:
    """Intersection over union of two half-open, millisecond intervals."""
    if start_a < 0 or start_b < 0 or end_a <= start_a or end_b <= start_b:
        raise ValueError("Intervals must have non-negative starts and positive duration")
    intersection = max(0, min(end_a, end_b) - max(start_a, start_b))
    union = max(end_a, end_b) - min(start_a, start_b)
    return intersection / union


@dataclass(slots=True)
class _Node:
    english_start: int
    english_end: int
    sorani_start: int
    sorani_end: int
    iou: float
    score: float
    covered: int
    merges: int
    previous: _Node | None


def _better(left: _Node | None, right: _Node | None) -> _Node | None:
    if left is None:
        return right
    if right is None:
        return left
    left_key = (round(left.score, 12), left.covered, -left.merges)
    right_key = (round(right.score, 12), right.covered, -right.merges)
    return left if left_key >= right_key else right


class _FenwickBest:
    """Best earlier alignment ending at or before a target-cue position."""

    def __init__(self, size: int) -> None:
        self._tree: list[_Node | None] = [None] * (size + 1)

    def update(self, position: int, node: _Node) -> None:
        # position is an exclusive end index, hence always at least 1.
        while position < len(self._tree):
            self._tree[position] = _better(self._tree[position], node)
            position += position & -position

    def query(self, position: int) -> _Node | None:
        result = None
        while position > 0:
            result = _better(result, self._tree[position])
            position -= position & -position
        return result


def _validate_cues(cues: list[Cue], language: str) -> None:
    seen: set[int] = set()
    for cue in cues:
        if cue.index in seen:
            raise ValueError(f"Duplicate {language} cue index: {cue.index}")
        seen.add(cue.index)
        if cue.start_ms < 0 or cue.end_ms <= cue.start_ms:
            raise ValueError(f"Invalid {language} cue timing at index {cue.index}")


def align_cues(
    english_cues: list[Cue],
    sorani_cues: list[Cue],
    *,
    min_iou: float = 0.5,
    tolerance_ms: int = 500,
    max_group_size: int = 3,
    max_candidates: int = 1_000_000,
) -> list[AlignedPair]:
    """Align cues one-to-one or merge consecutive cues on exactly one side.

    Matching is order-preserving; each cue is used at most once. The objective
    rewards IoU and coverage, with a small penalty for each merged cue. The
    tolerance applies to *each outer boundary*, not a timestamp shift or padded
    IoU. Only candidates whose unpadded IoU is at least ``min_iou`` are emitted.
    """
    if not 0 < min_iou <= 1:
        raise ValueError("min_iou must be in (0, 1]")
    if tolerance_ms < 0 or max_group_size < 1 or max_candidates < 1:
        raise ValueError("tolerance must be non-negative and group/candidate limits positive")
    _validate_cues(english_cues, "English")
    _validate_cues(sorani_cues, "Sorani")
    english = sorted(english_cues, key=lambda cue: (cue.start_ms, cue.index))
    sorani = sorted(sorani_cues, key=lambda cue: (cue.start_ms, cue.index))
    if not english or not sorani:
        return []

    starts = [cue.start_ms for cue in sorani]
    best_by_end = _FenwickBest(len(sorani))
    pending: list[tuple[int, int, _Node]] = []
    serial = 0
    best: _Node | None = None

    for i, first_en in enumerate(english):
        # Only groups ending before i can precede the candidates beginning at i.
        while pending and pending[0][0] <= i:
            _, _, completed = heapq.heappop(pending)
            best_by_end.update(completed.sorani_end, completed)

        lo = bisect.bisect_left(starts, first_en.start_ms - tolerance_ms)
        hi = bisect.bisect_right(starts, first_en.start_ms + tolerance_ms)
        for en_size in range(1, min(max_group_size, len(english) - i) + 1):
            en_end = max(cue.end_ms for cue in english[i : i + en_size])
            for j in range(lo, hi):
                max_ckb_size = min(max_group_size, len(sorani) - j) if en_size == 1 else 1
                for ckb_size in range(1, max_ckb_size + 1):
                    ckb_end = max(cue.end_ms for cue in sorani[j : j + ckb_size])
                    if abs(en_end - ckb_end) > tolerance_ms:
                        continue
                    iou = temporal_iou(first_en.start_ms, en_end, sorani[j].start_ms, ckb_end)
                    if iou < min_iou:
                        continue
                    serial += 1
                    if serial > max_candidates:
                        raise ValueError(
                            "Too many temporal alignment candidates; check for duplicated or "
                            "overlapping timestamps, or process a smaller file."
                        )
                    previous = best_by_end.query(j)
                    merge_count = en_size + ckb_size - 2
                    value = iou * (en_size + ckb_size) - 0.15 * merge_count
                    node = _Node(
                        english_start=i,
                        english_end=i + en_size,
                        sorani_start=j,
                        sorani_end=j + ckb_size,
                        iou=iou,
                        score=(previous.score if previous else 0.0) + value,
                        covered=(previous.covered if previous else 0) + en_size + ckb_size,
                        merges=(previous.merges if previous else 0) + merge_count,
                        previous=previous,
                    )
                    heapq.heappush(pending, (node.english_end, serial, node))
                    best = _better(best, node)

    result: list[AlignedPair] = []
    while best is not None:
        result.append(
            AlignedPair(
                tuple(english[best.english_start : best.english_end]),
                tuple(sorani[best.sorani_start : best.sorani_end]),
                best.iou,
            )
        )
        best = best.previous
    result.reverse()
    return result
