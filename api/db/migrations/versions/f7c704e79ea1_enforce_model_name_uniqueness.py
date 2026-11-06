"""enforce model name uniqueness

Revision ID: f7c704e79ea1
Revises: 72e3b67599cd
Created: 2026-11-06 19:08:07.241001
"""

import sqlalchemy as sa
from alembic import op

revision = "f7c704e79ea1"
down_revision = "72e3b67599cd"
branch_labels = None
depends_on = None


def upgrade():
    op.create_index(
        "uq_model_global_name_direction",
        "model_versions",
        ["name", "direction"],
        unique=True,
        postgresql_where=sa.text("owner_id IS NULL"),
        sqlite_where=sa.text("owner_id IS NULL"),
    )
    op.create_unique_constraint(
        "uq_model_owner_name_direction", "model_versions", ["owner_id", "name", "direction"]
    )


def downgrade():
    op.drop_constraint("uq_model_owner_name_direction", "model_versions", type_="unique")
    op.drop_index(
        "uq_model_global_name_direction",
        table_name="model_versions",
        postgresql_where=sa.text("owner_id IS NULL"),
        sqlite_where=sa.text("owner_id IS NULL"),
    )
