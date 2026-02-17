"""add_oauth_and_username_columns

Revision ID: b4576bc80a40
Revises: abfd982b29b2
Create Date: 2025-10-23 20:28:55.925065

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "b4576bc80a40"
down_revision = "abfd982b29b2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add OAuth and username columns to users table
    op.add_column(
        "users", sa.Column("oauth_provider", sa.String(length=50), nullable=True)
    )
    op.add_column("users", sa.Column("oauth_id", sa.String(length=255), nullable=True))
    op.add_column(
        "users", sa.Column("profile_picture", sa.String(length=500), nullable=True)
    )
    op.add_column("users", sa.Column("username", sa.String(length=100), nullable=True))

    # Create index on username for uniqueness
    op.create_index(op.f("ix_users_username"), "users", ["username"], unique=True)


def downgrade() -> None:
    # Drop columns in reverse order
    op.drop_index(op.f("ix_users_username"), table_name="users")
    op.drop_column("users", "username")
    op.drop_column("users", "profile_picture")
    op.drop_column("users", "oauth_id")
    op.drop_column("users", "oauth_provider")
