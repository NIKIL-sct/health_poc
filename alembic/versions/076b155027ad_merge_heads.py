"""merge heads

Revision ID: 076b155027ad
Revises: 2e2fd96653a5, add_camera_latency_table
Create Date: 2026-01-29 06:36:20.154000+00:00

"""

from alembic import op
import sqlalchemy as sa



# revision identifiers, used by Alembic.
revision = '076b155027ad'
down_revision = ('2e2fd96653a5', 'add_camera_latency_table')
branch_labels = None
depends_on = None


def upgrade():
    pass


def downgrade():
    pass
