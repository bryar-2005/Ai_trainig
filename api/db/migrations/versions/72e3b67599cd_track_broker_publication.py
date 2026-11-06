"""track broker publication

Revision ID: 72e3b67599cd
Revises: 490510cf55ab
Created: 2026-11-06 19:06:48.456527
"""

import sqlalchemy as sa
from alembic import op

revision = "72e3b67599cd"
down_revision = "490510cf55ab"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("jobs", sa.Column("enqueued_at", sa.DateTime(timezone=True), nullable=True))


def downgrade():
    op.drop_column("jobs", "enqueued_at")
