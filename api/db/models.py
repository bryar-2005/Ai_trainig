"""Persistent, owner-scoped project records and immutable review history."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[str]: JSON}


class User(Base):
    __tablename__ = "users"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(120))
    password_hash: Mapped[str] = mapped_column(String(512))
    role: Mapped[str] = mapped_column(String(16), default="user")
    preferences: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, onupdate=now)
    __table_args__ = (CheckConstraint("role IN ('admin','reviewer','user')", name="ck_user_role"),)


class RefreshSession(Base):
    __tablename__ = "refresh_sessions"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    rotated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class PersonalApiToken(Base):
    """A high-entropy bearer token; only its SHA-256 digest is persisted."""

    __tablename__ = "personal_api_tokens"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(120))
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class PasswordReset(Base):
    """Single-use, short-lived reset tokens, stored as digests only."""

    __tablename__ = "password_resets"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    owner_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True
    )
    name: Mapped[str] = mapped_column(String(150))
    direction: Mapped[str] = mapped_column(String(8))
    kind: Mapped[str] = mapped_column(String(16))
    # Trusted server-side paths or explicitly configured Hub model IDs, never uploaded code.
    source: Mapped[str] = mapped_column(Text)
    adapter_dir: Mapped[str | None] = mapped_column(Text, nullable=True)
    revision: Mapped[str | None] = mapped_column(String(40), nullable=True)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    __table_args__ = (
        CheckConstraint("direction IN ('en-ckb','ckb-en')", name="ck_model_direction"),
        CheckConstraint("kind IN ('base','adapter','merged')", name="ck_model_kind"),
        UniqueConstraint("owner_id", "name", "direction", name="uq_model_owner_name_direction"),
        Index(
            "uq_model_global_name_direction",
            "name",
            "direction",
            unique=True,
            postgresql_where=text("owner_id IS NULL"),
            sqlite_where=text("owner_id IS NULL"),
        ),
    )


class Project(Base):
    __tablename__ = "projects"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    owner_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    direction: Mapped[str] = mapped_column(String(8))
    status: Mapped[str] = mapped_column(String(24), default="draft")
    model_version_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("model_versions.id", ondelete="SET NULL"), nullable=True
    )
    target_cps: Mapped[float] = mapped_column(Float, default=17.0)
    hard_cps: Mapped[float] = mapped_column(Float, default=21.0)
    max_chars: Mapped[int] = mapped_column(Integer, default=42)
    max_lines: Mapped[int] = mapped_column(Integer, default=2)
    register: Mapped[str] = mapped_column(String(12), default="auto")
    soften_profanity: Mapped[bool] = mapped_column(Boolean, default=False)
    profanity_replacements: Mapped[list[dict[str, str]]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, onupdate=now)
    __table_args__ = (
        CheckConstraint("direction IN ('en-ckb','ckb-en')", name="ck_project_direction"),
        CheckConstraint(
            "status IN ('draft','translating','needs_review','completed','failed')",
            name="ck_project_status",
        ),
        CheckConstraint("target_cps > 0 AND hard_cps >= target_cps", name="ck_project_cps"),
        CheckConstraint("max_chars > 0 AND max_lines BETWEEN 1 AND 2", name="ck_project_lines"),
        CheckConstraint("register IN ('auto','formal','informal')", name="ck_project_register"),
    )


class MediaFile(Base):
    __tablename__ = "media_files"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[str] = mapped_column(String(16))
    storage_key: Mapped[str] = mapped_column(String(255), unique=True)
    original_name: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str] = mapped_column(String(120))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    sha256: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    __table_args__ = (
        UniqueConstraint("project_id", "kind", name="uq_project_media_kind"),
        CheckConstraint("kind IN ('subtitle','video')", name="ck_media_kind"),
        CheckConstraint("size_bytes > 0", name="ck_media_size"),
    )


class Cue(Base):
    __tablename__ = "cues"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    cue_index: Mapped[int] = mapped_column(Integer)
    original_start_ms: Mapped[int] = mapped_column(Integer)
    original_end_ms: Mapped[int] = mapped_column(Integer)
    start_ms: Mapped[int] = mapped_column(Integer)
    end_ms: Mapped[int] = mapped_column(Integer)
    source_cue_ids: Mapped[list[int]] = mapped_column(JSON, default=list)
    sentence_source_cue_ids: Mapped[list[int]] = mapped_column(JSON, default=list)
    source_offsets: Mapped[list[int] | None] = mapped_column(JSON, nullable=True)
    source_text: Mapped[str] = mapped_column(Text)
    normalized_source_text: Mapped[str] = mapped_column(Text, default="")
    speaker: Mapped[str | None] = mapped_column(String(120), nullable=True)
    speaker_labels: Mapped[list[str]] = mapped_column(JSON, default=list)
    sound_effects: Mapped[list[str]] = mapped_column(JSON, default=list)
    style: Mapped[str | None] = mapped_column(String(120), nullable=True)
    machine_text: Mapped[str] = mapped_column(Text, default="")
    human_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_version_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("model_versions.id", ondelete="SET NULL"), nullable=True
    )
    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    mean_token_logprob: Mapped[float | None] = mapped_column(Float, nullable=True)
    cps: Mapped[float] = mapped_column(Float, default=0.0)
    line_lengths: Mapped[list[int]] = mapped_column(JSON, default=list)
    length_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    glossary_mismatches: Mapped[list[str]] = mapped_column(JSON, default=list)
    qa_warnings: Mapped[list[str]] = mapped_column(JSON, default=list)
    flagged: Mapped[bool] = mapped_column(Boolean, default=True)
    status: Mapped[str] = mapped_column(String(16), default="unreviewed")
    revision: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, onupdate=now)
    __table_args__ = (
        UniqueConstraint("project_id", "cue_index", name="uq_project_cue_index"),
        CheckConstraint("cue_index > 0 AND revision >= 0", name="ck_cue_index_revision"),
        CheckConstraint("start_ms >= 0 AND end_ms > start_ms", name="ck_cue_time"),
        CheckConstraint(
            "original_start_ms >= 0 AND original_end_ms > original_start_ms",
            name="ck_cue_original_time",
        ),
        CheckConstraint("status IN ('unreviewed','edited','approved')", name="ck_cue_status"),
        Index("ix_cue_project_status", "project_id", "status"),
    )


class CueRevision(Base):
    __tablename__ = "cue_revisions"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    cue_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("cues.id", ondelete="CASCADE"), index=True
    )
    actor_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("users.id", ondelete="CASCADE"))
    revision: Mapped[int] = mapped_column(Integer)
    reason: Mapped[str] = mapped_column(String(40))
    before: Mapped[dict[str, Any]] = mapped_column(JSON)
    after: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    __table_args__ = (UniqueConstraint("cue_id", "revision", name="uq_cue_revision"),)


class GlossaryTerm(Base):
    __tablename__ = "glossary_terms"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    english: Mapped[str] = mapped_column(String(300))
    english_key: Mapped[str] = mapped_column(String(300))
    sorani: Mapped[str] = mapped_column(String(300))
    category: Mapped[str] = mapped_column(String(40), default="unknown")
    locked: Mapped[bool] = mapped_column(Boolean, default=True)
    origin: Mapped[str] = mapped_column(String(24), default="user")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, onupdate=now)
    __table_args__ = (UniqueConstraint("project_id", "english_key", name="uq_glossary_english"),)


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    owner_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("projects.id", ondelete="CASCADE"), nullable=True, index=True
    )
    kind: Mapped[str] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(24), default="queued", index=True)
    stage: Mapped[str] = mapped_column(String(80), default="queued")
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    enqueued_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    lease_token: Mapped[uuid.UUID | None] = mapped_column(Uuid, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, onupdate=now)
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued','running','cancel_requested','cancelled','completed','failed')",
            name="ck_job_status",
        ),
        CheckConstraint(
            "kind IN ('translate','retranslate','train','compare','sample')", name="ck_job_kind"
        ),
        CheckConstraint("progress >= 0 AND progress <= 100", name="ck_job_progress"),
    )


class JobEvent(Base):
    __tablename__ = "job_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("jobs.id", ondelete="CASCADE"), index=True
    )
    stage: Mapped[str] = mapped_column(String(80))
    progress: Mapped[float] = mapped_column(Float)
    message: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class ReviewEvent(Base):
    __tablename__ = "review_events"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    cue_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("cues.id", ondelete="CASCADE"), index=True
    )
    actor_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("users.id", ondelete="CASCADE"))
    action: Mapped[str] = mapped_column(String(24))
    cue_revision: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
