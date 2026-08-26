"""
Service configuration, read from the environment.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def _normalise_database_url(url: str) -> str:
    """
    Points bare PostgreSQL URLs at psycopg 3.

    Hosting platforms hand out connection strings as postgres:// or
    postgresql://, and SQLAlchemy maps both to psycopg2 - a driver this project
    does not install. Rewriting here means a deployment can paste the string the
    platform gave it and have the service work.
    """
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


# SQLite by default so the service runs with no infrastructure; Compose and any
# real deployment point this at PostgreSQL.
DATABASE_URL = _normalise_database_url(
    os.getenv("DATABASE_URL", f"sqlite:///{PROJECT_ROOT / 'decisions.db'}")
)

# Deliberately no default. An unset key makes the protected endpoints fail
# closed with 503 rather than silently serving credit decisions to anyone.
API_KEY = os.getenv("API_KEY", "")

API_KEY_HEADER = "X-API-Key"

# Page size limits for the decision log.
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 200
