"""fence worker leases

Revision ID: fef5a853934e
Revises: f7c704e79ea1
Created: 2026-11-06 19:25:15.781558
"""

import sqlalchemy as sa
from alembic import op

revision = "fef5a853934e"
down_revision = "f7c704e79ea1"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("jobs", sa.Column("lease_token", sa.Uuid(), nullable=True))


def downgrade():
    op.drop_column("jobs", "lease_token")
