"""Allow authenticated, model-backed sample translation jobs.

Revision ID: 0d72530d4840
Revises: 942953576e51
"""

import sqlalchemy as sa
from alembic import op

revision = "0d72530d4840"
down_revision = "942953576e51"
branch_labels = None
depends_on = None


def upgrade():
    op.drop_constraint("ck_job_kind", "jobs", type_="check")
    op.create_check_constraint(
        "ck_job_kind", "jobs", "kind IN ('translate','retranslate','train','compare','sample')"
    )


def downgrade():
    # Preserve user data: an operator must explicitly remove sample jobs first.
    count = op.get_bind().execute(sa.text("SELECT count(*) FROM jobs WHERE kind = 'sample'"))
    if count.scalar_one():
        raise RuntimeError("Delete or archive sample jobs before downgrading this migration")
    op.drop_constraint("ck_job_kind", "jobs", type_="check")
    op.create_check_constraint(
        "ck_job_kind", "jobs", "kind IN ('translate','retranslate','train','compare')"
    )
