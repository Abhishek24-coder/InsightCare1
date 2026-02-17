# alembic/env.py (snippet)
from logging.config import fileConfig
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # project root
from sqlalchemy import engine_from_config, pool
from alembic import context
from app.database import Base
from app.config import DATABASE_URL

config = context.config
fileConfig(config.config_file_name)
target_metadata = Base.metadata


def run_migrations_online():
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    # optionally override url with env
    connectable = connectable.execution_options(isolation_level="AUTOCOMMIT")
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
