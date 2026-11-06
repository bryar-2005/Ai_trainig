"""Quality filters, opt-in fastText verification, and bilingual MinHash deduplication."""

from __future__ import annotations

import math
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasketch import MinHash, MinHashLSH

ENGLISH_LABELS = frozenset({"en", "eng", "eng_latn"})
SORANI_LABELS = frozenset({"ckb", "ckb_arab", "ckbarab"})


@dataclass(frozen=True, slots=True)
class LanguagePrediction:
    label: str
    confidence: float


def _canonical_label(label: str) -> str:
    return label.removeprefix("__label__").casefold().replace("-", "_")


class FastTextLanguageIdentifier:
    """Local fastText classifier with distinct English and Sorani labels.

    Script alone cannot distinguish Sorani from Arabic or Persian. Generic
    ``ku`` is not accepted as verified Sorani, even when the text is Arabic.
    """

    def __init__(self, model_path: str | Path, *, min_confidence: float = 0.3) -> None:
        if not 0 <= min_confidence <= 1:
            raise ValueError("Language ID confidence threshold must be in [0, 1]")
        path = Path(model_path)
        if not path.is_file():
            raise ValueError(f"Language ID model not found: {path}")
        try:
            import fasttext
        except ImportError as exc:
            raise ValueError("Install the project's fasttext-wheel dependency to use LID") from exc
        try:
            self._model = fasttext.load_model(str(path))
            labels = {_canonical_label(label) for label in self._model.get_labels()}
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError(f"Could not load fastText model {path}: {exc}") from exc
        if not (labels & ENGLISH_LABELS) or not (labels & SORANI_LABELS):
            raise ValueError(
                "The fastText model must contain both English (en/eng) and explicit Sorani "
                "(ckb/ckb_Arab) labels; generic ku alone is not sufficient."
            )
        self.min_confidence = min_confidence
        self._cache: OrderedDict[str, LanguagePrediction] = OrderedDict()

    def predict(self, text: str) -> LanguagePrediction:
        text = " ".join(text.split())
        cached = self._cache.get(text)
        if cached is not None:
            self._cache.move_to_end(text)
            return cached
        labels, probabilities = self._model.predict(text, k=1)
        probability = float(probabilities[0])
        if not math.isfinite(probability):
            raise ValueError("Language ID model returned a non-finite probability")
        # fastText may add a tiny epsilon to probabilities.
        result = LanguagePrediction(_canonical_label(labels[0]), max(0.0, min(1.0, probability)))
        self._cache[text] = result
        if len(self._cache) > 100_000:
            self._cache.popitem(last=False)
        return result


@dataclass(frozen=True, slots=True)
class FilterResult:
    accepted: bool
    reason: str | None
    length_ratio: float
    script_ratio: float
    english_language: LanguagePrediction | None = None
    sorani_language: LanguagePrediction | None = None


def arabic_script_ratio(text: str) -> float:
    """Arabic-script letters divided by all letters; digits/punctuation are ignored."""
    letters = [char for char in text if unicodedata.category(char).startswith("L")]
    if not letters:
        return 0.0
    arabic = sum("ARABIC" in unicodedata.name(char, "") for char in letters)
    return arabic / len(letters)


def filter_pair(
    english: str,
    sorani: str,
    *,
    detector: FastTextLanguageIdentifier | None,
    min_script_ratio: float = 0.8,
    min_length_ratio: float = 0.4,
    max_length_ratio: float = 2.5,
) -> FilterResult:
    """Filter a normalized bilingual pair; ``detector=None`` is unverified mode."""
    if not 0 <= min_script_ratio <= 1 or not 0 < min_length_ratio <= max_length_ratio:
        raise ValueError("Invalid script or length ratio thresholds")
    en_len = len("".join(english.split()))
    ckb_len = len("".join(sorani.split()))
    ratio = ckb_len / en_len if en_len else 0.0
    script = arabic_script_ratio(sorani)

    def reject(reason: str) -> FilterResult:
        return FilterResult(False, reason, ratio, script)

    if not en_len or not ckb_len:
        return reject("empty_text")
    if " ".join(english.casefold().split()) == " ".join(sorani.casefold().split()):
        return reject("identical_text")
    if not min_length_ratio <= ratio <= max_length_ratio:
        return reject("length_ratio")
    if script < min_script_ratio:
        return reject("sorani_script_ratio")
    if detector is None:
        return FilterResult(True, None, ratio, script)
    en_prediction = detector.predict(english)
    ckb_prediction = detector.predict(sorani)
    reason = None
    if en_prediction.label not in ENGLISH_LABELS:
        reason = "english_language_id"
    elif ckb_prediction.label not in SORANI_LABELS:
        reason = "sorani_language_id"
    elif min(en_prediction.confidence, ckb_prediction.confidence) < detector.min_confidence:
        reason = "language_id_low_confidence"
    return FilterResult(reason is None, reason, ratio, script, en_prediction, ckb_prediction)


def _canonical_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _shingles(text: str) -> set[str]:
    text = f"^{_canonical_text(text)}$"
    return {text[i : i + 3] for i in range(max(1, len(text) - 2))}


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def deduplicate_pairs(
    rows: list[dict[str, Any]], *, threshold: float = 0.90
) -> tuple[list[dict], int]:
    """Use MinHash LSH plus exact Jaccard checks on *each* language.

    Exact bilingual duplicates are always removed. Near-duplicate candidate
    retrieval is probabilistic, as with any LSH; exact per-language similarity
    is checked before removal to avoid discarding distinct translations.
    Human-labeled records win over unknown and machine records.
    """
    if not 0 < threshold <= 1:
        raise ValueError("Deduplication threshold must be in (0, 1]")
    provenance_rank = {"human": 0, "unknown": 1, "machine": 2}
    ordered = sorted(
        rows,
        key=lambda row: (
            provenance_rank.get(row["provenance"], 1),
            -row["quality_score"],
            row["film_id"],
            row["english_start_ms"],
            row["pair_id"],
        ),
    )
    # 32 bands x 8 rows of 256 permutations gives high recall near 0.90;
    # exact Jaccard checks below remove false positives.
    index = MinHashLSH(threshold=threshold, num_perm=256, params=(32, 8))
    kept: list[dict[str, Any]] = []
    kept_shingles: list[tuple[set[str], set[str]]] = []
    exact: set[tuple[str, str]] = set()
    removed = 0
    for row in ordered:
        pair = (_canonical_text(row["english"]), _canonical_text(row["sorani"]))
        if pair in exact:
            removed += 1
            continue
        en_shingles = _shingles(pair[0])
        ckb_shingles = _shingles(pair[1])
        signature = MinHash(num_perm=256, seed=1)
        tokens = [f"en:{token}".encode() for token in en_shingles]
        tokens += [f"ckb:{token}".encode() for token in ckb_shingles]
        signature.update_batch(tokens)
        duplicate = False
        for candidate in sorted(index.query(signature), key=int):
            old_en, old_ckb = kept_shingles[int(candidate)]
            if (
                _jaccard(en_shingles, old_en) >= threshold
                and _jaccard(ckb_shingles, old_ckb) >= threshold
            ):
                duplicate = True
                break
        if duplicate:
            removed += 1
            continue
        key = str(len(kept))
        index.insert(key, signature)
        exact.add(pair)
        kept.append(row)
        kept_shingles.append((en_shingles, ckb_shingles))
    kept.sort(key=lambda row: (row["film_id"], row["english_start_ms"], row["pair_id"]))
    return kept, removed
