"""create place_crowding_snapshots

Revision ID: c1b2a3d4e5f6
Revises: 59085e4c2fa3
Create Date: 2026-01-13
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "c1b2a3d4e5f6"
down_revision = "59085e4c2fa3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "place_crowding_snapshots",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),

        sa.Column("place_id", sa.String(length=50), nullable=False),
        sa.Column("place_name", sa.String(length=200), nullable=True),
        sa.Column("category_group_code", sa.String(length=10), nullable=True),

        sa.Column("lat", sa.Float(), nullable=True),
        sa.Column("lng", sa.Float(), nullable=True),

        sa.Column("zone_code", sa.String(length=20), nullable=False),
        sa.Column("zone_distance_m", sa.Float(), nullable=True),

        sa.Column("ts", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),

        sa.Column("level", sa.String(length=50), nullable=False, server_default=""),
        sa.Column("rank", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("message", sa.String(length=500), nullable=False, server_default=""),
        sa.Column("updated_at_epoch", sa.BigInteger(), nullable=False, server_default="0"),

        sa.Column("raw", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )

    op.create_index(
        "ix_place_crowding_snapshots_place_id_ts",
        "place_crowding_snapshots",
        ["place_id", "ts"],
    )
    op.create_index(
        "ix_place_crowding_snapshots_zone_code_ts",
        "place_crowding_snapshots",
        ["zone_code", "ts"],
    )


def downgrade() -> None:
    op.drop_index("ix_place_crowding_snapshots_zone_code_ts", table_name="place_crowding_snapshots")
    op.drop_index("ix_place_crowding_snapshots_place_id_ts", table_name="place_crowding_snapshots")
    op.drop_table("place_crowding_snapshots")
