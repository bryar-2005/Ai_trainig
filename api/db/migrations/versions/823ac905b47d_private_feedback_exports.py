"""Track private feedback snapshots and deduplicate nightly schedules.

Revision ID: 823ac905b47d
Revises: 0d72530d4840
"""

import sqlalchemy as sa
from alembic import op

revision = "823ac905b47d"
down_revision = "0d72530d4840"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("jobs", sa.Column("schedule_key", sa.String(100), nullable=True))
    op.create_unique_constraint("uq_jobs_schedule_key", "jobs", ["schedule_key"])
    op.drop_constraint("ck_job_kind", "jobs", type_="check")
    op.create_check_constraint(
        "ck_job_kind",
        "jobs",
        "kind IN ('translate','retranslate','train','compare','sample','feedback_export')",
    )
    op.create_table(
        "feedback_exports",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("owner_id", sa.Uuid(), nullable=False),
        sa.Column("job_id", sa.Uuid(), nullable=False),
        sa.Column("lease_token", sa.Uuid(), nullable=False),
        sa.Column("storage_key", sa.String(255), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("direction", sa.String(8), nullable=True),
        sa.Column("cue_count", sa.Integer(), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("status IN ('pending','ready')", name="ck_feedback_export_status"),
        sa.CheckConstraint(
            "direction IS NULL OR direction IN ('en-ckb','ckb-en')",
            name="ck_feedback_export_direction",
        ),
        sa.CheckConstraint("cue_count > 0 AND size_bytes > 0", name="ck_feedback_export_size"),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("storage_key"),
    )
    for column in ("owner_id", "job_id", "expires_at"):
        op.create_index(f"ix_feedback_exports_{column}", "feedback_exports", [column])


def downgrade():
    bind = op.get_bind()
    if bind.execute(sa.text("SELECT count(*) FROM feedback_exports")).scalar_one():
        raise RuntimeError("Delete or archive private feedback exports before downgrading")
    if bind.execute(
        sa.text("SELECT count(*) FROM jobs WHERE kind = 'feedback_export'")
    ).scalar_one():
        raise RuntimeError("Delete or archive feedback export jobs before downgrading")
    for column in ("expires_at", "job_id", "owner_id"):
        op.drop_index(f"ix_feedback_exports_{column}", table_name="feedback_exports")
    op.drop_table("feedback_exports")
    op.drop_constraint("ck_job_kind", "jobs", type_="check")
    op.create_check_constraint(
        "ck_job_kind", "jobs", "kind IN ('translate','retranslate','train','compare','sample')"
    )
    op.drop_constraint("uq_jobs_schedule_key", "jobs", type_="unique")
    op.drop_column("jobs", "schedule_key")
