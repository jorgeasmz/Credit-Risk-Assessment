from alembic.config import Config
from sqlalchemy import create_engine, inspect

from alembic import command
from app import settings


def test_migrations_build_the_schema_from_empty(tmp_path, monkeypatch):
    """
    Runs the real migrations against an empty database.

    Without this the schema could drift from the models and nobody would notice
    until a deployment tried to upgrade a production database.
    """
    url = f"sqlite:///{tmp_path / 'migrated.db'}"
    monkeypatch.setattr(settings, "DATABASE_URL", url)

    config = Config(str(settings.PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(settings.PROJECT_ROOT / "alembic"))
    command.upgrade(config, "head")

    inspector = inspect(create_engine(url))
    assert "decisions" in inspector.get_table_names()

    columns = {c["name"] for c in inspector.get_columns("decisions")}
    assert {"model_version", "threshold", "contributions", "defaulted"} <= columns
