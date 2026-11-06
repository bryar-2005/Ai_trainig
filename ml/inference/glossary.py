"""Film-wide named-entity proposals and auditable locked terminology."""

from __future__ import annotations

import json
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import regex

from ml.data.normalize import normalize_english, normalize_sorani
from ml.inference.model_source import resolve_model_source
from ml.inference.text import _BIDI


@dataclass(frozen=True, slots=True)
class GlossaryEntry:
    english: str
    sorani: str
    locked: bool = True
    category: str = "unknown"
    origin: str = "user"


@dataclass(frozen=True, slots=True)
class NamedEntity:
    text: str
    category: str = "unknown"


class EntityExtractor(ABC):
    @abstractmethod
    def extract(self, texts: list[str]) -> list[NamedEntity]:
        """Extract English entity candidates across the supplied script."""


_WORDS_TO_SKIP = frozenset(
    "a an and are as at be because but by can could did do does for from get go good "
    "he hello her here hey hi his how i if in is it its let me my no not now of oh on "
    "or our please she so that the their then there these they this to uh um us we well "
    "what when where which who why will with yes you your".split()
)
_NAME_WORD = r"(?:[\p{Latin}&&\p{Lu}][[\p{Latin}&&\p{Ll}]\p{M}]{1,}|[\p{Latin}&&\p{Lu}]{2,8})"
_PROPER_NAME = regex.compile(
    rf"(?V1)(?<![\p{{L}}\p{{N}}_]){_NAME_WORD}(?:\s+{_NAME_WORD}){{0,3}}(?![\p{{L}}\p{{N}}_])"
)


class HeuristicEntityExtractor(EntityExtractor):
    """Conservative capitalized-name candidates; category and spelling need review."""

    def extract(self, texts: list[str]) -> list[NamedEntity]:
        found: dict[str, NamedEntity] = {}
        for text in texts:
            for match in _PROPER_NAME.finditer(text):
                name = match.group().strip()
                words = name.split()
                # Common sentence-initial words are not entity evidence.
                while words and words[0].casefold() in _WORDS_TO_SKIP:
                    words.pop(0)
                if not words or all(word.casefold() in _WORDS_TO_SKIP for word in words):
                    continue
                name = " ".join(words)
                found.setdefault(name.casefold(), NamedEntity(name))
        return list(found.values())


class TransformersEntityExtractor(EntityExtractor):
    """Optional local or explicitly configured Hugging Face token-classification NER."""

    def __init__(self, model: str, *, local_files_only: bool = False, device: str = "cpu"):
        if not model or device not in {"cpu", "cuda"}:
            raise ValueError("NER requires a model and cpu or cuda device")
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        self.model_source = model
        source, cached_revision = resolve_model_source(model, local_files_only=local_files_only)
        self.resolved_source = source
        # A string passed directly to pipeline can trigger Hub lookups before
        # model_kwargs.local_files_only is forwarded. Load explicit objects.
        ner_model = AutoModelForTokenClassification.from_pretrained(
            source, local_files_only=local_files_only, trust_remote_code=False
        )
        revision = getattr(ner_model.config, "_commit_hash", None) or cached_revision
        self.model_revision = revision
        ner_tokenizer = AutoTokenizer.from_pretrained(
            source,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=False,
        )
        self._pipeline = pipeline(
            "token-classification",
            model=ner_model,
            tokenizer=ner_tokenizer,
            aggregation_strategy="simple",
            device=0 if device == "cuda" else -1,
        )
        tokenizer = self._pipeline.tokenizer
        if not tokenizer.is_fast:
            raise ValueError("The NER tokenizer must be fast to support overlapping long inputs")
        limit = int(tokenizer.model_max_length)
        position_limit = getattr(self._pipeline.model.config, "max_position_embeddings", None)
        if isinstance(position_limit, int) and position_limit > 0:
            limit = min(limit, position_limit)
        if limit < 8:
            raise ValueError("The NER model's maximum sequence length is too small")
        tokenizer.model_max_length = limit
        self._stride = max(1, min(32, (limit - 2) // 4))

    def extract(self, texts: list[str]) -> list[NamedEntity]:
        if not texts:
            return []
        found: dict[tuple[str, str], NamedEntity] = {}
        for text in texts:
            for result in self._pipeline(text, stride=self._stride):
                label = (
                    str(result.get("entity_group", "")).upper().replace("B-", "").replace("I-", "")
                )
                category = {
                    "PER": "person",
                    "PERSON": "person",
                    "LOC": "place",
                    "LOCATION": "place",
                    "GPE": "place",
                    "ORG": "organization",
                    "ORGANIZATION": "organization",
                }.get(label)
                start, end = result.get("start"), result.get("end")
                if not category or not isinstance(start, int) or not isinstance(end, int):
                    continue
                name = " ".join(text[start:end].split())
                if name:
                    found.setdefault((name.casefold(), category), NamedEntity(name, category))
        return list(found.values())


_DIGRAPHS = {
    "tch": "چ",
    "sh": "ش",
    "ch": "چ",
    "zh": "ژ",
    "kh": "خ",
    "gh": "غ",
    "ph": "ف",
    "th": "ث",
    "oo": "وو",
    "ee": "ی",
    "ou": "او",
    "qu": "کو",
}
_LETTERS = {
    "a": "ا",
    "b": "ب",
    "c": "ک",
    "d": "د",
    "e": "ە",
    "f": "ف",
    "g": "گ",
    "h": "ھ",
    "i": "ی",
    "j": "ج",
    "k": "ک",
    "l": "ل",
    "m": "م",
    "n": "ن",
    "o": "ۆ",
    "p": "پ",
    "q": "ق",
    "r": "ر",
    "s": "س",
    "t": "ت",
    "u": "و",
    "v": "ڤ",
    "w": "و",
    "x": "کس",
    "y": "ی",
    "z": "ز",
}


def propose_transliteration(name: str) -> str:
    """Produce a deterministic, approximate proposal, never an authoritative spelling."""
    result: list[str] = []
    # Approximate common accented Latin letters before applying the simple
    # English spelling map. These are proposals, never locked spellings.
    decomposed = unicodedata.normalize("NFKD", name.casefold())
    text = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    text = text.translate(str.maketrans({"ł": "l", "ø": "o", "đ": "d", "ð": "d", "þ": "th"}))
    index = 0
    while index < len(text):
        match = next((part for part in _DIGRAPHS if text.startswith(part, index)), None)
        if match:
            result.append(_DIGRAPHS[match])
            index += len(match)
        else:
            result.append(_LETTERS.get(text[index], text[index]))
            index += 1
    return "".join(result).strip()


def _normalized(text: str) -> str:
    return " ".join(
        unicodedata.normalize("NFC", "".join(c for c in text if c not in _BIDI)).casefold().split()
    )


def _pattern(term: str) -> regex.Pattern:
    return regex.compile(
        r"(?<![\p{L}\p{N}\p{M}_])" + regex.escape(_normalized(term)) + r"(?![\p{L}\p{N}\p{M}_])",
        regex.IGNORECASE,
    )


def contains_term(text: str, term: str) -> bool:
    return bool(_pattern(term).search(_normalized(text)))


def load_glossary(path: str | Path) -> list[GlossaryEntry]:
    try:
        items = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read glossary: {exc}") from exc
    if not isinstance(items, list):
        raise ValueError("Glossary must be a JSON array")
    result: list[GlossaryEntry] = []
    seen: set[str] = set()
    for index, item in enumerate(items, 1):
        if not isinstance(item, dict):
            raise ValueError(f"Glossary item {index} must be an object")
        english, sorani = item.get("english"), item.get("sorani")
        locked, category = item.get("locked", True), item.get("category", "unknown")
        if (
            not isinstance(english, str)
            or not english.strip()
            or not isinstance(sorani, str)
            or not sorani.strip()
        ):
            raise ValueError(f"Glossary item {index} requires English and Sorani text")
        if not isinstance(locked, bool) or not isinstance(category, str) or not category.strip():
            raise ValueError(f"Glossary item {index} has invalid lock or category")
        key = _normalized(english)
        if key in seen:
            raise ValueError(f"Glossary English term {english!r} occurs more than once")
        seen.add(key)
        result.append(GlossaryEntry(english.strip(), sorani.strip(), locked, category.strip()))
    return result


def merge_entities(terms: list[GlossaryEntry], entities: list[NamedEntity]) -> list[GlossaryEntry]:
    """Keep normalized user entries authoritative and append reviewable proposals."""
    merged: list[GlossaryEntry] = []
    known: set[str] = set()
    for term in terms:
        if (
            not isinstance(term, GlossaryEntry)
            or not isinstance(term.english, str)
            or not isinstance(term.sorani, str)
        ):
            raise ValueError("Glossary entries require English and Sorani strings")
        english, sorani = normalize_english(term.english), normalize_sorani(term.sorani)
        if not english or not sorani or not isinstance(term.locked, bool):
            raise ValueError("Glossary entries require nonempty terms and a boolean lock")
        if not isinstance(term.category, str) or not isinstance(term.origin, str):
            raise ValueError("Glossary category and origin must be strings")
        key = _normalized(english)
        if key in known:
            raise ValueError(f"Glossary English term {english!r} occurs more than once")
        known.add(key)
        merged.append(GlossaryEntry(english, sorani, term.locked, term.category, term.origin))
    for entity in entities:
        key = _normalized(entity.text)
        if not key or key in known:
            continue
        target = propose_transliteration(entity.text)
        if target:
            merged.append(
                GlossaryEntry(entity.text, target, False, entity.category, "ner_proposal")
            )
            known.add(key)
    return merged


def relevant_locked(source: str, terms: list[GlossaryEntry], direction: str) -> list[GlossaryEntry]:
    if direction not in {"en-ckb", "ckb-en"}:
        raise ValueError("Unsupported translation direction")
    return [
        term
        for term in terms
        if term.locked
        and contains_term(source, term.english if direction == "en-ckb" else term.sorani)
    ]


def missing_locked(text: str, terms: list[GlossaryEntry], direction: str) -> list[GlossaryEntry]:
    return [
        term
        for term in terms
        if not contains_term(text, term.sorani if direction == "en-ckb" else term.english)
    ]


def replace_literal_sources(text: str, missing: list[GlossaryEntry], direction: str) -> str:
    """Replace a still-untranslated source term; never insert a term at a guessed location."""
    result = text
    for term in sorted(
        missing, key=lambda item: -len(item.english if direction == "en-ckb" else item.sorani)
    ):
        source, target = (
            (term.english, term.sorani) if direction == "en-ckb" else (term.sorani, term.english)
        )
        # Preserve other content and original case. Whitespace must match exactly;
        # a failed replacement remains an explicit glossary QA warning.
        pattern = regex.compile(
            r"(?<![\p{L}\p{N}\p{M}_])" + regex.escape(source) + r"(?![\p{L}\p{N}\p{M}_])",
            regex.IGNORECASE,
        )
        result = pattern.sub(lambda _, replacement=target: replacement, result)
    return result
