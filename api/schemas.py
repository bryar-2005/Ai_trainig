"""Strict API request contracts; clients cannot set owners or privileged roles."""

from __future__ import annotations

import math
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator, model_validator


class InputModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @field_validator("*", mode="before")
    @classmethod
    def strip_input(cls, value, info):
        # Passwords and human subtitle edits are literal user input.
        if isinstance(value, str) and info.field_name not in {"password", "human_text"}:
            return value.strip()
        return value


class RegisterIn(InputModel):
    email: EmailStr
    display_name: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=12, max_length=128)


class LoginIn(InputModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class SofteningRule(InputModel):
    source: str = Field(min_length=1, max_length=120)
    replacement: str = Field(min_length=1, max_length=120)

    @model_validator(mode="after")
    def single_line(self):
        if any(char in text for text in (self.source, self.replacement) for char in ("\n", "\r")):
            raise ValueError("Softening terms must be single-line text")
        return self


class ProjectCreate(InputModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=5000)
    direction: Literal["en-ckb", "ckb-en"]
    model_version_id: uuid.UUID | None = None
    target_cps: float = Field(default=17, gt=0, le=100, allow_inf_nan=False)
    hard_cps: float = Field(default=21, gt=0, le=100, allow_inf_nan=False)
    max_chars: int = Field(default=42, ge=1, le=200)
    max_lines: int = Field(default=2, ge=1, le=2)
    register_mode: Literal["auto", "formal", "informal"] = Field(default="auto", alias="register")
    soften_profanity: bool = False
    profanity_replacements: list[SofteningRule] = Field(default_factory=list, max_length=50)

    @model_validator(mode="after")
    def cps_order(self):
        if self.target_cps > self.hard_cps:
            raise ValueError("target_cps must not exceed hard_cps")
        if self.soften_profanity and not self.profanity_replacements:
            raise ValueError("Provide explicit target-language softening replacements")
        keys = [rule.source.casefold() for rule in self.profanity_replacements]
        if len(keys) != len(set(keys)):
            raise ValueError("Softening source terms must be unique")
        return self


class ProjectPatch(InputModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=5000)
    model_version_id: uuid.UUID | None = None
    target_cps: float | None = Field(default=None, gt=0, le=100, allow_inf_nan=False)
    hard_cps: float | None = Field(default=None, gt=0, le=100, allow_inf_nan=False)
    max_chars: int | None = Field(default=None, ge=1, le=200)
    max_lines: int | None = Field(default=None, ge=1, le=2)
    register_mode: Literal["auto", "formal", "informal"] | None = Field(
        default=None, alias="register"
    )
    soften_profanity: bool | None = None
    profanity_replacements: list[SofteningRule] | None = Field(default=None, max_length=50)


class CuePatch(InputModel):
    expected_revision: int = Field(ge=0)
    # Null clears the human edit; omission leaves it untouched.
    human_text: str | None = Field(default=None, max_length=10000)
    start_ms: int | None = Field(default=None, ge=0, le=2_147_483_647)
    end_ms: int | None = Field(default=None, ge=1, le=2_147_483_647)
    status: Literal["unreviewed", "edited", "approved"] | None = None


class RevisionRequest(InputModel):
    expected_revision: int = Field(ge=0)


class BulkApproval(InputModel):
    cues: list[tuple[uuid.UUID, int]] = Field(min_length=1, max_length=1000)

    @field_validator("cues")
    @classmethod
    def unique_nonnegative(cls, value):
        if len({item[0] for item in value}) != len(value) or any(v < 0 for _, v in value):
            raise ValueError("Each cue must occur once with a nonnegative revision")
        return value


class GlossaryIn(InputModel):
    english: str = Field(min_length=1, max_length=300)
    sorani: str = Field(min_length=1, max_length=300)
    category: str = Field(default="unknown", min_length=1, max_length=40)
    locked: bool = True


class GlossaryPatch(InputModel):
    english: str | None = Field(default=None, min_length=1, max_length=300)
    sorani: str | None = Field(default=None, min_length=1, max_length=300)
    category: str | None = Field(default=None, min_length=1, max_length=40)
    locked: bool | None = None


class TranslationIn(InputModel):
    model_version_id: uuid.UUID | None = None
    allow_replace_machine: bool = False


class RetranslateIn(InputModel):
    expected_revision: int = Field(ge=0)
    custom_context: list[str] = Field(default_factory=list, max_length=5)
    glossary_hint: GlossaryIn | None = None

    @field_validator("custom_context")
    @classmethod
    def short_context(cls, value):
        if any(not item.strip() or len(item) > 2000 for item in value):
            raise ValueError("Context sentences must be nonempty and at most 2000 characters")
        return value


class ModelCreate(InputModel):
    name: str = Field(min_length=1, max_length=150)
    direction: Literal["en-ckb", "ckb-en"]
    kind: Literal["base", "adapter", "merged"]
    source: str = Field(min_length=1, max_length=1024)
    adapter_dir: str | None = Field(default=None, max_length=1024)
    revision: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    metrics: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_model(self):
        if (self.kind == "adapter") != bool(self.adapter_dir):
            raise ValueError("adapter_dir is required only for an adapter")
        if self.kind == "adapter" and self.revision:
            raise ValueError("Adapters use the base revision in their training manifest")
        if any(not math.isfinite(value) for value in self.metrics.values()):
            raise ValueError("Metrics must be finite")
        return self


class TrainIn(InputModel):
    direction: Literal["en-ckb", "ckb-en"]
    model_version_id: uuid.UUID
    epochs: float = Field(default=3, gt=0, le=10, allow_inf_nan=False)
    learning_rate: float = Field(default=2e-4, gt=0, le=1e-2, allow_inf_nan=False)
    # Human approval verifies the edit, not automatic language ID.
    allow_unverified_language: bool = False


class SampleIn(InputModel):
    """A small, authenticated interactive sample; output comes from a registered model."""

    direction: Literal["en-ckb", "ckb-en"]
    model_version_id: uuid.UUID
    lines: list[str] = Field(min_length=1, max_length=3)

    @field_validator("lines")
    @classmethod
    def bounded_text(cls, value):
        if any(not item.strip() or len(item) > 2000 for item in value):
            raise ValueError("Lines must be nonempty and at most 2000 characters")
        return value


class CompareIn(InputModel):
    direction: Literal["en-ckb", "ckb-en"]
    model_a: uuid.UUID
    model_b: uuid.UUID
    lines: list[str] = Field(min_length=1, max_length=5)
    context: list[str] = Field(default_factory=list, max_length=5)

    @field_validator("lines", "context")
    @classmethod
    def bounded_text(cls, value):
        if any(not item.strip() or len(item) > 2000 for item in value):
            raise ValueError("Lines and context must be nonempty and at most 2000 characters")
        return value

    @model_validator(mode="after")
    def distinct_models(self):
        if self.model_a == self.model_b:
            raise ValueError("Choose two different model versions")
        return self
