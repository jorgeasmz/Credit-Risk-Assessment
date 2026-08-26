import logging
import secrets
from typing import Annotated

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app import settings

logger = logging.getLogger(__name__)

# auto_error=False so a missing header reaches the handler below and gets the
# same treatment as a wrong one, instead of a framework-shaped 403.
_scheme = APIKeyHeader(name=settings.API_KEY_HEADER, auto_error=False)


def require_api_key(provided: Annotated[str | None, Security(_scheme)]) -> None:
    """
    Guards the endpoints that score applicants or read the decision log.

    A single shared key is the right shape for service-to-service access, which
    is what this API is for; there are no user sessions to model. A deployment
    with several consumers would need per-client keys, rotation and rate limits,
    and that is a different design rather than a bigger version of this one.
    """
    if not settings.API_KEY:
        logger.error("API_KEY is unset; refusing protected requests.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API key is not configured on this server.",
        )

    # Constant-time comparison: a plain == leaks the key one byte at a time
    # through timing.
    if not provided or not secrets.compare_digest(provided, settings.API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
