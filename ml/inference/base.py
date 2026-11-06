"""Model-independent batched translation interface and result types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class TranslationRequest:
    """A sentence, its previous source sentences, and applicable term hints."""

    text: str
    context: tuple[str, ...] = ()
    speaker: str | None = None
    glossary_hints: tuple[tuple[str, str], ...] = ()
    force_terms: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TranslationResult:
    """Generated text and normalized selected-token log probabilities."""

    text: str
    token_logprobs: tuple[float, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseTranslator(ABC):
    """Swap the underlying model without coupling the pipeline to a framework."""

    model_version_id: str
    direction: str

    @abstractmethod
    def translate_batch(self, requests: list[TranslationRequest]) -> list[TranslationResult]:
        """Translate in order, respecting supported hard ``force_terms`` constraints."""
