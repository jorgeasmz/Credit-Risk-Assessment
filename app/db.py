from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.settings import DATABASE_URL


class Base(DeclarativeBase):
    """Declarative base for the ORM models."""


def _engine_options(url: str) -> dict:
    """
    SQLite needs one concession that PostgreSQL does not.

    FastAPI serves synchronous endpoints from a threadpool, and SQLite refuses
    to reuse a connection across threads unless told otherwise.
    """
    if url.startswith("sqlite"):
        return {"connect_args": {"check_same_thread": False}}
    return {"pool_pre_ping": True}


engine = create_engine(DATABASE_URL, **_engine_options(DATABASE_URL))
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_session() -> Iterator[Session]:
    """FastAPI dependency yielding a session that always gets closed."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
