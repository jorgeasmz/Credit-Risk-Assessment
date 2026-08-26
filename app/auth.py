import logging
import secrets
from typing import Annotated

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app import settings

logger = logging.getLogger(__name__)

# auto_error=False so a missing header is handled here, not as a framework 403.
_scheme = APIKeyHeader(name=settings.API_KEY_HEADER, auto_error=False)


def require_api_key(provided: Annotated[str | None, Security(_scheme)]) -> None:
    """Rejects any request that does not carry the configured API key."""
    if not settings.API_KEY:
        logger.error("API_KEY is unset; refusing protected requests.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API key is not configured on this server.",
        )

    # Constant-time: a plain == leaks the key through timing.
    if not provided or not secrets.compare_digest(provided, settings.API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
