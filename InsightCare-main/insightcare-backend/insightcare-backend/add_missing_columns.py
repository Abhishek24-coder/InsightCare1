"""
Add missing columns to users table
Run this script to fix the database schema
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from sqlalchemy import text
from app.database import engine


def add_missing_columns():
    """Add missing columns to users table"""

    columns_to_add = [
        ("is_verified", "BOOLEAN DEFAULT FALSE"),
        ("verification_token", "VARCHAR(255)"),
        ("verification_token_expires", "TIMESTAMP"),
        ("reset_token", "VARCHAR(255)"),
        ("reset_token_expires", "TIMESTAMP"),
    ]

    with engine.connect() as conn:
        print("🔧 Checking and adding missing columns...\n")

        for column_name, column_def in columns_to_add:
            try:
                # Try to add the column
                sql = f"ALTER TABLE users ADD COLUMN {column_name} {column_def};"
                conn.execute(text(sql))
                conn.commit()
                print(f"✅ Added column: {column_name}")
            except Exception as e:
                if "already exists" in str(e) or "duplicate column" in str(e).lower():
                    print(f"ℹ️  Column {column_name} already exists")
                else:
                    print(f"⚠️  Error adding {column_name}: {e}")

        print("\n✅ Database schema updated successfully!")
        print("🚀 You can now login/register")


if __name__ == "__main__":
    add_missing_columns()
