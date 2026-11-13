"""Deterministic, bounded text-overlap signals for the offline review audit.

These are character-level hints, not proof of leakage, independence, language,
translation quality, or rights. The caller is responsible for validating and
protecting the input records and for checking work identities separately.
"""

from __future__ import annotations

import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field

# An exact match is always checked. Near matches are intentionally limited to
# longer strings and to a single, documented character-trigram set metric.
NEAR_MIN_CHARS = 12
_NEAR_NUMERATOR = 9
_NEAR_DENOMINATOR = 10


class OverlapLimitError(ValueError):
    """The bounded audit cannot finish without exceeding a resource limit."""


@dataclass(slots=True)
class WorkBudget:
    """Stop rather than silently skipping candidates or returning a partial audit."""

    max_reference_windows: int = 2_000_000
    max_query_windows: int = 100_000_000
    max_posting_visits: int = 20_000_000
    max_candidate_checks: int = 5_000_000
    reference_windows: int = 0
    query_windows: int = 0
    posting_visits: int = 0
    candidate_checks: int = 0

    def spend(self, kind: str, amount: int) -> None:
        current = getattr(self, kind) + amount
        if current > getattr(self, f"max_{kind}"):
            raise OverlapLimitError(
                f"Overlap audit exceeded its {kind.replace('_', ' ')} limit; "
                "no complete report was published"
            )
        setattr(self, kind, current)


def canonical_text(text: str) -> str:
    """NFKC, Unicode casefold, and whitespace collapse; retain ZWNJ and punctuation."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _shingles(text: str, budget: WorkBudget, *, reference: bool) -> frozenset[str] | None:
    if len(text) < NEAR_MIN_CHARS:
        return None
    # Charge the number of windows before constructing a set, even when many
    # windows repeat. This also bounds work for highly repetitive long text.
    windows = len(text) - 2
    budget.spend("reference_windows" if reference else "query_windows", windows)
    return frozenset(text[index : index + 3] for index in range(windows))


def _prefix(shingles: frozenset[str], frequency: Counter[str]) -> list[str]:
    """Prefix filter with a consistent ordering; never drops a >=0.9 Jaccard hit.

    If J(A,B) >= 0.9, their intersection is at least ceil(0.9*|A|) and
    ceil(0.9*|B|). The smallest common shingle must therefore appear within
    these prefixes of *both* sets. Exact integer arithmetic avoids rounding at
    the threshold. The later intersection/union check removes false positives.
    Frequency ordering is fixed by the reference split, not by each query.
    """
    count = len(shingles)
    required = (_NEAR_NUMERATOR * count + _NEAR_DENOMINATOR - 1) // _NEAR_DENOMINATOR
    ordered = sorted(shingles, key=lambda token: (frequency.get(token, 0), token))
    return ordered[: count - required + 1]


def _near_score(left: frozenset[str], right: frozenset[str]) -> float | None:
    small, large = sorted((len(left), len(right)))
    if small * _NEAR_DENOMINATOR < large * _NEAR_NUMERATOR:
        return None
    intersection = len(left & right)
    union = len(left) + len(right) - intersection
    if intersection * _NEAR_DENOMINATOR < union * _NEAR_NUMERATOR:
        return None
    return intersection / union


@dataclass(frozen=True, slots=True)
class AuditPair:
    """One validated pair, with text already canonicalized by the caller."""

    pair_id: str
    english: str
    sorani: str
    film_id: str | None = None
    source_group_id: str | None = None


@dataclass(frozen=True, slots=True)
class QueryPair:
    pair: AuditPair
    english_shingles: frozenset[str] | None
    sorani_shingles: frozenset[str] | None


def prepare_query(pair: AuditPair, budget: WorkBudget) -> QueryPair:
    return QueryPair(
        pair,
        _shingles(pair.english, budget, reference=False),
        _shingles(pair.sorani, budget, reference=False),
    )


@dataclass(slots=True)
class PairMatch:
    signals: set[str] = field(default_factory=set)
    similarities: dict[str, float] = field(default_factory=dict)


class ReferenceIndex:
    """Index one held-out split/variant, including one-sided and swapped matches."""

    def __init__(self, records: list[AuditPair], budget: WorkBudget) -> None:
        self.records = sorted(records, key=lambda record: record.pair_id)
        self.by_id: dict[str, list[int]] = defaultdict(list)
        self.by_pair: dict[tuple[str, str], list[int]] = defaultdict(list)
        self.exact: dict[str, dict[str, list[int]]] = {
            "english": defaultdict(list),
            "sorani": defaultdict(list),
        }
        self.prefixes: dict[str, dict[str, list[int]]] = {
            "english": defaultdict(list),
            "sorani": defaultdict(list),
        }
        self.shingles: dict[str, list[frozenset[str] | None]] = {
            "english": [],
            "sorani": [],
        }
        self.frequency: dict[str, Counter[str]] = {"english": Counter(), "sorani": Counter()}
        for index, record in enumerate(self.records):
            self.by_id[record.pair_id].append(index)
            self.by_pair[(record.english, record.sorani)].append(index)
            for language in ("english", "sorani"):
                text = getattr(record, language)
                self.exact[language][text].append(index)
                tokens = _shingles(text, budget, reference=True)
                self.shingles[language].append(tokens)
                if tokens is not None:
                    self.frequency[language].update(tokens)
        # Rare-first ordering reduces candidate explosions without sampling.
        # Both sides use the same order, including zero-frequency query tokens.
        for language in ("english", "sorani"):
            for index, tokens in enumerate(self.shingles[language]):
                if tokens is not None:
                    for token in _prefix(tokens, self.frequency[language]):
                        self.prefixes[language][token].append(index)

    def find(self, query: QueryPair, budget: WorkBudget) -> list[tuple[AuditPair, PairMatch]]:
        matches: dict[int, PairMatch] = {}

        def add_many(indices: list[int], signal: str, language: str | None = None) -> None:
            budget.spend("posting_visits", len(indices))
            for index in indices:
                match = matches.setdefault(index, PairMatch())
                match.signals.add(signal)
                if language:
                    match.similarities[language] = 1.0

        pair = query.pair
        add_many(self.by_id.get(pair.pair_id, []), "same_pair_id")
        for language in ("english", "sorani"):
            text = getattr(pair, language)
            add_many(self.exact[language].get(text, []), f"{language}_exact", language)
        # Standardized English/Sorani fields make ordinary reverse-direction
        # training an ordinary exact match. This extra signal catches swapped
        # field labels; equal fields cannot establish a swapped orientation.
        if pair.english != pair.sorani:
            add_many(self.by_pair.get((pair.sorani, pair.english), []), "swapped_bilingual_exact")

        for language in ("english", "sorani"):
            query_tokens = getattr(query, f"{language}_shingles")
            if query_tokens is None:
                continue
            candidates: set[int] = set()
            for token in _prefix(query_tokens, self.frequency[language]):
                postings = self.prefixes[language].get(token, [])
                budget.spend("posting_visits", len(postings))
                candidates.update(postings)
            budget.spend("candidate_checks", len(candidates))
            for index in sorted(candidates):
                reference = self.records[index]
                if getattr(reference, language) == getattr(pair, language):
                    continue  # Already counted as exact, not as a near match.
                reference_tokens = self.shingles[language][index]
                assert reference_tokens is not None
                score = _near_score(query_tokens, reference_tokens)
                if score is not None:
                    match = matches.setdefault(index, PairMatch())
                    match.signals.add(f"{language}_near")
                    match.similarities[language] = round(score, 6)

        # Also bound huge one-sided exact-match Cartesian products.
        budget.spend("candidate_checks", len(matches))
        result = []
        for index in sorted(matches):
            match = matches[index]
            en = match.signals & {"english_exact", "english_near"}
            ckb = match.signals & {"sorani_exact", "sorani_near"}
            if en and ckb:
                if {"english_exact", "sorani_exact"} <= match.signals:
                    match.signals.add("bilingual_exact")
                else:
                    match.signals.add("bilingual_near")
            result.append((self.records[index], match))
        return result
