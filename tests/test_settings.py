import pytest

from app.settings import _normalise_database_url


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        # What Render and Heroku actually hand out.
        ("postgres://u:p@host/db", "postgresql+psycopg://u:p@host/db"),
        ("postgresql://u:p@host/db", "postgresql+psycopg://u:p@host/db"),
        # Already explicit, and non-PostgreSQL URLs, are left alone.
        ("postgresql+psycopg://u:p@host/db", "postgresql+psycopg://u:p@host/db"),
        ("sqlite:///decisions.db", "sqlite:///decisions.db"),
    ],
)
def test_database_urls_are_pointed_at_psycopg_3(given, expected):
    """SQLAlchemy maps a bare postgresql:// to psycopg2, which is not installed."""
    assert _normalise_database_url(given) == expected


def test_a_password_containing_the_scheme_is_not_mangled():
    url = "postgresql://user:postgres://weird@host/db"

    assert _normalise_database_url(url) == "postgresql+psycopg://user:postgres://weird@host/db"
