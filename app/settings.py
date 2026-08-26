"""
Service configuration, read from the environment.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def _normalise_database_url(url: str) -> str:
    """Points bare postgres:// and postgresql:// URLs at psycopg 3."""
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


# SQLite by default; Compose and real deployments point this at PostgreSQL.
DATABASE_URL = _normalise_database_url(
    os.getenv("DATABASE_URL", f"sqlite:///{PROJECT_ROOT / 'decisions.db'}")
)

# No default: an unset key makes the protected endpoints fail closed.
API_KEY = os.getenv("API_KEY", "")

API_KEY_HEADER = "X-API-Key"

# Buckets used for the score histogram.
SCORE_BUCKETS = 10

# Page size limits for the decision log.
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 200
