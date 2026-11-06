"""initial project and review schema

Revision ID: 490510cf55ab
Revises:
Created: 2026-11-06 18:57:41.126004
"""

import sqlalchemy as sa
from alembic import op

revision = "490510cf55ab"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("display_name", sa.String(length=120), nullable=False),
        sa.Column("password_hash", sa.String(length=512), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("preferences", sa.JSON(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("role IN ('admin','reviewer','user')", name="ck_user_role"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_users_email"), "users", ["email"], unique=True)
    op.create_table(
        "model_versions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("owner_id", sa.Uuid(), nullable=True),
        sa.Column("name", sa.String(length=150), nullable=False),
        sa.Column("direction", sa.String(length=8), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("adapter_dir", sa.Text(), nullable=True),
        sa.Column("revision", sa.String(length=40), nullable=True),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("direction IN ('en-ckb','ckb-en')", name="ck_model_direction"),
        sa.CheckConstraint("kind IN ('base','adapter','merged')", name="ck_model_kind"),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_model_versions_owner_id"), "model_versions", ["owner_id"], unique=False
    )
    op.create_table(
        "refresh_sessions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("rotated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("token_hash"),
    )
    op.create_index(
        op.f("ix_refresh_sessions_user_id"), "refresh_sessions", ["user_id"], unique=False
    )
    op.create_table(
        "projects",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("owner_id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("direction", sa.String(length=8), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("model_version_id", sa.Uuid(), nullable=True),
        sa.Column("target_cps", sa.Float(), nullable=False),
        sa.Column("hard_cps", sa.Float(), nullable=False),
        sa.Column("max_chars", sa.Integer(), nullable=False),
        sa.Column("max_lines", sa.Integer(), nullable=False),
        sa.Column("register", sa.String(length=12), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("direction IN ('en-ckb','ckb-en')", name="ck_project_direction"),
        sa.CheckConstraint("register IN ('auto','formal','informal')", name="ck_project_register"),
        sa.CheckConstraint(
            "status IN ('draft','translating','needs_review','completed','failed')",
            name="ck_project_status",
        ),
        sa.CheckConstraint("max_chars > 0 AND max_lines BETWEEN 1 AND 2", name="ck_project_lines"),
        sa.CheckConstraint("target_cps > 0 AND hard_cps >= target_cps", name="ck_project_cps"),
        sa.ForeignKeyConstraint(["model_version_id"], ["model_versions.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_projects_owner_id"), "projects", ["owner_id"], unique=False)
    op.create_table(
        "cues",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("cue_index", sa.Integer(), nullable=False),
        sa.Column("original_start_ms", sa.Integer(), nullable=False),
        sa.Column("original_end_ms", sa.Integer(), nullable=False),
        sa.Column("start_ms", sa.Integer(), nullable=False),
        sa.Column("end_ms", sa.Integer(), nullable=False),
        sa.Column("source_cue_ids", sa.JSON(), nullable=False),
        sa.Column("sentence_source_cue_ids", sa.JSON(), nullable=False),
        sa.Column("source_offsets", sa.JSON(), nullable=True),
        sa.Column("source_text", sa.Text(), nullable=False),
        sa.Column("normalized_source_text", sa.Text(), nullable=False),
        sa.Column("speaker", sa.String(length=120), nullable=True),
        sa.Column("speaker_labels", sa.JSON(), nullable=False),
        sa.Column("sound_effects", sa.JSON(), nullable=False),
        sa.Column("style", sa.String(length=120), nullable=True),
        sa.Column("machine_text", sa.Text(), nullable=False),
        sa.Column("human_text", sa.Text(), nullable=True),
        sa.Column("model_version_id", sa.Uuid(), nullable=True),
        sa.Column("confidence_score", sa.Float(), nullable=True),
        sa.Column("mean_token_logprob", sa.Float(), nullable=True),
        sa.Column("cps", sa.Float(), nullable=False),
        sa.Column("line_lengths", sa.JSON(), nullable=False),
        sa.Column("length_ratio", sa.Float(), nullable=True),
        sa.Column("glossary_mismatches", sa.JSON(), nullable=False),
        sa.Column("qa_warnings", sa.JSON(), nullable=False),
        sa.Column("flagged", sa.Boolean(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("status IN ('unreviewed','edited','approved')", name="ck_cue_status"),
        sa.CheckConstraint("cue_index > 0 AND revision >= 0", name="ck_cue_index_revision"),
        sa.CheckConstraint(
            "original_start_ms >= 0 AND original_end_ms > original_start_ms",
            name="ck_cue_original_time",
        ),
        sa.CheckConstraint("start_ms >= 0 AND end_ms > start_ms", name="ck_cue_time"),
        sa.ForeignKeyConstraint(["model_version_id"], ["model_versions.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("project_id", "cue_index", name="uq_project_cue_index"),
    )
    op.create_index("ix_cue_project_status", "cues", ["project_id", "status"], unique=False)
    op.create_index(op.f("ix_cues_project_id"), "cues", ["project_id"], unique=False)
    op.create_table(
        "glossary_terms",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("english", sa.String(length=300), nullable=False),
        sa.Column("english_key", sa.String(length=300), nullable=False),
        sa.Column("sorani", sa.String(length=300), nullable=False),
        sa.Column("category", sa.String(length=40), nullable=False),
        sa.Column("locked", sa.Boolean(), nullable=False),
        sa.Column("origin", sa.String(length=24), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("project_id", "english_key", name="uq_glossary_english"),
    )
    op.create_index(
        op.f("ix_glossary_terms_project_id"), "glossary_terms", ["project_id"], unique=False
    )
    op.create_table(
        "jobs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("owner_id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=True),
        sa.Column("kind", sa.String(length=20), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("stage", sa.String(length=80), nullable=False),
        sa.Column("progress", sa.Float(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "kind IN ('translate','retranslate','train','compare')", name="ck_job_kind"
        ),
        sa.CheckConstraint(
            "status IN ('queued','running','cancel_requested','cancelled','completed','failed')",
            name="ck_job_status",
        ),
        sa.CheckConstraint("progress >= 0 AND progress <= 100", name="ck_job_progress"),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_jobs_owner_id"), "jobs", ["owner_id"], unique=False)
    op.create_index(op.f("ix_jobs_project_id"), "jobs", ["project_id"], unique=False)
    op.create_index(op.f("ix_jobs_status"), "jobs", ["status"], unique=False)
    op.create_table(
        "media_files",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("storage_key", sa.String(length=255), nullable=False),
        sa.Column("original_name", sa.String(length=255), nullable=False),
        sa.Column("content_type", sa.String(length=120), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("kind IN ('subtitle','video')", name="ck_media_kind"),
        sa.CheckConstraint("size_bytes > 0", name="ck_media_size"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("project_id", "kind", name="uq_project_media_kind"),
        sa.UniqueConstraint("storage_key"),
    )
    op.create_index(op.f("ix_media_files_project_id"), "media_files", ["project_id"], unique=False)
    op.create_table(
        "cue_revisions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("cue_id", sa.Uuid(), nullable=False),
        sa.Column("actor_id", sa.Uuid(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("reason", sa.String(length=40), nullable=False),
        sa.Column("before", sa.JSON(), nullable=False),
        sa.Column("after", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["actor_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["cue_id"], ["cues.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("cue_id", "revision", name="uq_cue_revision"),
    )
    op.create_index(op.f("ix_cue_revisions_cue_id"), "cue_revisions", ["cue_id"], unique=False)
    op.create_table(
        "job_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("job_id", sa.Uuid(), nullable=False),
        sa.Column("stage", sa.String(length=80), nullable=False),
        sa.Column("progress", sa.Float(), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_job_events_job_id"), "job_events", ["job_id"], unique=False)
    op.create_table(
        "review_events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("cue_id", sa.Uuid(), nullable=False),
        sa.Column("actor_id", sa.Uuid(), nullable=False),
        sa.Column("action", sa.String(length=24), nullable=False),
        sa.Column("cue_revision", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["actor_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["cue_id"], ["cues.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_review_events_cue_id"), "review_events", ["cue_id"], unique=False)


def downgrade():
    op.drop_index(op.f("ix_review_events_cue_id"), table_name="review_events")
    op.drop_table("review_events")
    op.drop_index(op.f("ix_job_events_job_id"), table_name="job_events")
    op.drop_table("job_events")
    op.drop_index(op.f("ix_cue_revisions_cue_id"), table_name="cue_revisions")
    op.drop_table("cue_revisions")
    op.drop_index(op.f("ix_media_files_project_id"), table_name="media_files")
    op.drop_table("media_files")
    op.drop_index(op.f("ix_jobs_status"), table_name="jobs")
    op.drop_index(op.f("ix_jobs_project_id"), table_name="jobs")
    op.drop_index(op.f("ix_jobs_owner_id"), table_name="jobs")
    op.drop_table("jobs")
    op.drop_index(op.f("ix_glossary_terms_project_id"), table_name="glossary_terms")
    op.drop_table("glossary_terms")
    op.drop_index(op.f("ix_cues_project_id"), table_name="cues")
    op.drop_index("ix_cue_project_status", table_name="cues")
    op.drop_table("cues")
    op.drop_index(op.f("ix_projects_owner_id"), table_name="projects")
    op.drop_table("projects")
    op.drop_index(op.f("ix_refresh_sessions_user_id"), table_name="refresh_sessions")
    op.drop_table("refresh_sessions")
    op.drop_index(op.f("ix_model_versions_owner_id"), table_name="model_versions")
    op.drop_table("model_versions")
    op.drop_index(op.f("ix_users_email"), table_name="users")
    op.drop_table("users")
