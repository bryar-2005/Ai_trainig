"""Add opt-in, user-specified target-language softening rules.

Revision ID: 3e29ff2cfe17
Revises: fef5a853934e
"""

import sqlalchemy as sa
from alembic import op

revision = "3e29ff2cfe17"
down_revision = "fef5a853934e"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "projects",
        sa.Column("soften_profanity", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        "projects",
        sa.Column("profanity_replacements", sa.JSON(), nullable=False, server_default="[]"),
    )
    op.alter_column("projects", "soften_profanity", server_default=None)
    op.alter_column("projects", "profanity_replacements", server_default=None)


def downgrade():
    op.drop_column("projects", "profanity_replacements")
    op.drop_column("projects", "soften_profanity")
