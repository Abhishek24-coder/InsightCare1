"""add_oauth_fields_to_users

Revision ID: abfd982b29b2
Revises: 29be0aa277d3
Create Date: 2025-10-22 23:53:24.616074

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "abfd982b29b2"
down_revision = "29be0aa277d3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add OAuth fields to users table
    op.add_column(
        "users", sa.Column("oauth_provider", sa.String(length=50), nullable=True)
    )
    op.add_column("users", sa.Column("oauth_id", sa.String(length=255), nullable=True))
    op.add_column(
        "users", sa.Column("profile_picture", sa.String(length=500), nullable=True)
    )
    op.add_column("users", sa.Column("username", sa.String(length=100), nullable=True))

    # Create unique index on username
    op.create_index("ix_users_username", "users", ["username"], unique=True)


def downgrade() -> None:
    # Remove OAuth fields from users table
    op.drop_index("ix_users_username", table_name="users")
    op.drop_column("users", "username")
    op.drop_column("users", "profile_picture")
    op.drop_column("users", "oauth_id")
    op.drop_column("users", "oauth_provider")
