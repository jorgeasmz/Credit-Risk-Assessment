"""
Alembic environment.

The URL and metadata come from the application rather than from alembic.ini, so
there is a single source of truth for both and no connection string committed to
the repository.
"""

from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

# Imported for its side effect: the models must be registered on Base.metadata
# before autogenerate can see them.
import app.models  # noqa: F401
from alembic import context
from app.db import Base
from app.settings import DATABASE_URL

config = context.config
config.set_main_option("sqlalchemy.url", DATABASE_URL)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # SQLite cannot ALTER a column in place; batch mode rewrites the
            # table instead, so the same migrations run on both backends.
            render_as_batch=connection.dialect.name == "sqlite",
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
