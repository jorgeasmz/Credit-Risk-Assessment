import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from sqlalchemy.orm import Session

from app import repository
from app.auth import require_api_key
from app.db import get_session
from app.schemas import (
    CreditApplication,
    DecisionPage,
    DecisionRecord,
    OutcomeRequest,
    PortfolioSummary,
    PredictionResponse,
)
from app.settings import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from app.utils import Scorer, load_scorer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads the model at startup; a failure degrades the service instead of killing it."""
    try:
        app.state.scorer = load_scorer()
    except Exception:
        logger.exception("Model failed to load; scoring will return 503.")
        app.state.scorer = None

    yield

    app.state.scorer = None


app = FastAPI(
    title="German Credit Risk API",
    description=(
        "Scores credit applications, explains each decision and keeps an "
        "auditable record of every one it has made."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# Set before startup so the health check works even if lifespan never ran.
app.state.scorer = None


def get_scorer(request: Request) -> Scorer:
    """Dependency that yields the loaded model or fails the request."""
    scorer = request.app.state.scorer
    if scorer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Check the server logs.",
        )
    return scorer


ScorerDep = Annotated[Scorer, Depends(get_scorer)]
SessionDep = Annotated[Session, Depends(get_session)]

# Everything that scores an applicant or reads the log sits behind the key.
PROTECTED = [Depends(require_api_key)]


@app.get("/")
def root(request: Request):
    """Health check. Reports whether a model is loaded and which one."""
    scorer = request.app.state.scorer
    return {
        "message": "Credit Risk Prediction API is operational. Visit /docs.",
        "model_loaded": scorer is not None,
        "model_version": scorer.version if scorer is not None else None,
    }


@app.post("/predict", response_model=PredictionResponse, dependencies=PROTECTED)
def predict(application: CreditApplication, scorer: ScorerDep, session: SessionDep):
    """Scores one application, explains it and records the decision."""
    payload = application.model_dump()

    try:
        result = scorer.score(payload)
    except Exception:
        # Logged in full server-side; the client gets no internals.
        logger.exception("Scoring failed.")
        raise HTTPException(status_code=500, detail="Prediction failed.") from None

    decision = repository.record_decision(
        session,
        model_version=result["model_version"],
        threshold=result["threshold"],
        risk_class=result["risk_class"],
        probability=result["probability"],
        application=payload,
        contributions=result["contributions"],
    )

    return PredictionResponse(decision_id=decision.id, **result)


@app.get("/decisions", response_model=DecisionPage, dependencies=PROTECTED)
def list_decisions(
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
    cursor: Annotated[int | None, Query(ge=1)] = None,
):
    """The decision log, newest first, paginated by key."""
    items, next_cursor = repository.list_decisions(session, limit=limit, cursor=cursor)
    return DecisionPage(items=items, next_cursor=next_cursor)


@app.get("/decisions/{decision_id}", response_model=DecisionRecord, dependencies=PROTECTED)
def get_decision(decision_id: int, session: SessionDep):
    """Reproduces one past decision in full."""
    decision = repository.get_decision(session, decision_id)
    if decision is None:
        raise HTTPException(status_code=404, detail="Decision not found.")
    return decision


@app.post(
    "/decisions/{decision_id}/outcome",
    response_model=DecisionRecord,
    dependencies=PROTECTED,
)
def record_outcome(decision_id: int, outcome: OutcomeRequest, session: SessionDep):
    """Attaches the ground truth to a decision once the loan resolves."""
    decision = repository.record_outcome(
        session, decision_id, defaulted=outcome.defaulted
    )
    if decision is None:
        raise HTTPException(status_code=404, detail="Decision not found.")
    return decision


@app.get("/summary", response_model=PortfolioSummary, dependencies=PROTECTED)
def summary(session: SessionDep):
    """Aggregate figures over everything the service has decided."""
    return repository.summarise(session)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
