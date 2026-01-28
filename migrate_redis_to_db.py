"""add connectivity metrics support

Revision ID: connectivity_metrics_001
Revises: 2e2fd96653a5
Create Date: 2026-01-28 00:00:00.000000

This migration adds support for enhanced network connectivity metrics:
- No schema changes required (using existing meta_data JSON field)
- This is a placeholder migration to document the feature addition
- All metrics are stored in the meta_data JSONB field of health_logs table

The existing schema already supports:
- health_logs.meta_data (JSONB) - stores all metrics
- health_logs.event_type - supports new "CONNECTIVITY_CHECK" type
- health_logs.status - supports PASS/FAIL/WARNING statuses

New metrics stored in meta_data:
- packet_loss_percent (float)
- rtt_min_ms, rtt_avg_ms, rtt_max_ms (float)
- connection_success_rate (float)
- latency_min_ms, latency_avg_ms, latency_max_ms (float)
- overall_status (str): HEALTHY/DEGRADED/SERVICE_DOWN/PING_BLOCKED/DOWN
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'connectivity_metrics_001'
down_revision = '2e2fd96653a5'
branch_labels = None
depends_on = None


def upgrade():
    """
    No schema changes needed - existing meta_data JSONB field handles all metrics.
    This migration documents the feature addition for tracking purposes.
    """
    pass


def downgrade():
    """
    No schema changes to revert.
    Metrics data will remain in meta_data field if migration is rolled back.
    """
    pass