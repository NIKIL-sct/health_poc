"""
Alembic Migration: Add camera_latency table for connectivity checks

Revision ID: add_camera_latency_table
Create Date: 2026-01-29

"""

from alembic import op
import sqlalchemy as sa
from datetime import datetime


# revision identifiers, used by Alembic.
revision = 'add_camera_latency_table'
down_revision = None  # UPDATE THIS with your last migration ID
branch_labels = None
depends_on = None


def upgrade():
    """
    Create camera_latency table for storing connectivity metrics:
    - Latency measurements (RTT avg/min/max)
    - Packet loss statistics
    """
    op.create_table(
        'camera_latency',
        sa.Column('latency_id', sa.String(36), nullable=False),
        sa.Column('camera_id', sa.String(64), nullable=False),
        sa.Column('rtt_avg', sa.Float(), nullable=True, comment='Average Round Trip Time in ms'),
        sa.Column('rtt_min', sa.Float(), nullable=True, comment='Minimum Round Trip Time in ms'),
        sa.Column('rtt_max', sa.Float(), nullable=True, comment='Maximum Round Trip Time in ms'),
        sa.Column('packet_loss_percent', sa.Float(), nullable=True, comment='Packet loss percentage'),
        sa.Column('packets_sent', sa.Integer(), nullable=True, comment='Number of packets sent'),
        sa.Column('packets_received', sa.Integer(), nullable=True, comment='Number of packets received'),
        sa.Column('is_reachable', sa.Boolean(), nullable=True, default=True, comment='Whether camera is reachable'),
        sa.Column('check_type', sa.String(20), nullable=True, default='PING', comment='Type of connectivity check'),
        sa.Column('error_message', sa.Text(), nullable=True, comment='Error message if check failed'),
        sa.Column('timestamp', sa.DateTime(), nullable=False, default=datetime.utcnow),
        sa.PrimaryKeyConstraint('latency_id')
    )
    
    # Create indexes for better query performance
    op.create_index('ix_camera_latency_camera_id', 'camera_latency', ['camera_id'])
    op.create_index('ix_camera_latency_timestamp', 'camera_latency', ['timestamp'])
    op.create_index('ix_camera_latency_camera_timestamp', 'camera_latency', ['camera_id', 'timestamp'])


def downgrade():
    """
    Drop camera_latency table
    """
    op.drop_index('ix_camera_latency_camera_timestamp', table_name='camera_latency')
    op.drop_index('ix_camera_latency_timestamp', table_name='camera_latency')
    op.drop_index('ix_camera_latency_camera_id', table_name='camera_latency')
    op.drop_table('camera_latency')