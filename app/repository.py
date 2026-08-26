from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Decision
from model.config import COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE


def record_decision(
    session: Session,
    *,
    model_version: str,
    threshold: float,
    risk_class: int,
    probability: float,
    application: dict,
    contributions: dict,
) -> Decision:
    """Persists one scoring decision and returns it with its assigned id."""
    decision = Decision(
        model_version=model_version,
        threshold=threshold,
        risk_class=risk_class,
        probability=probability,
        application=application,
        contributions=contributions,
    )
    session.add(decision)
    session.commit()
    session.refresh(decision)
    return decision


def get_decision(session: Session, decision_id: int) -> Decision | None:
    """Single entry of the audit log, or None if the id is unknown."""
    return session.get(Decision, decision_id)


def list_decisions(
    session: Session, *, limit: int, cursor: int | None = None
) -> tuple[list[Decision], int | None]:
    """
    Newest first, paginated by key rather than by offset.

    OFFSET has to walk the rows it skips, so it degrades as the log grows, and
    it silently shifts entries when new decisions arrive mid-pagination. Seeking
    on the primary key has neither problem.
    """
    statement = select(Decision).order_by(Decision.id.desc()).limit(limit + 1)
    if cursor is not None:
        statement = statement.where(Decision.id < cursor)

    rows = list(session.scalars(statement))
    has_more = len(rows) > limit
    page = rows[:limit]
    next_cursor = page[-1].id if has_more and page else None
    return page, next_cursor


def record_outcome(
    session: Session, decision_id: int, *, defaulted: bool
) -> Decision | None:
    """Attaches the ground truth to a past decision. Returns None if unknown."""
    decision = session.get(Decision, decision_id)
    if decision is None:
        return None

    decision.defaulted = int(defaulted)
    decision.outcome_recorded_at = datetime.now(timezone.utc)
    session.commit()
    session.refresh(decision)
    return decision


def summarise(session: Session) -> dict:
    """
    Portfolio-level figures for the decision log.

    Realised cost is reported over the decisions that have an outcome, and only
    those: the cost matrix needs to know what actually happened, so a service
    without a feedback loop cannot measure what it is costing.
    """
    total = session.scalar(select(func.count(Decision.id))) or 0
    approved = (
        session.scalar(
            select(func.count(Decision.id)).where(Decision.risk_class == 0)
        )
        or 0
    )
    with_outcome = (
        session.scalar(
            select(func.count(Decision.id)).where(Decision.defaulted.is_not(None))
        )
        or 0
    )

    false_negatives = (
        session.scalar(
            select(func.count(Decision.id)).where(
                Decision.risk_class == 0, Decision.defaulted == 1
            )
        )
        or 0
    )
    false_positives = (
        session.scalar(
            select(func.count(Decision.id)).where(
                Decision.risk_class == 1, Decision.defaulted == 0
            )
        )
        or 0
    )

    return {
        "total": total,
        "approved": approved,
        "rejected": total - approved,
        "approval_rate": approved / total if total else 0.0,
        "mean_probability": float(session.scalar(select(func.avg(Decision.probability))) or 0.0),
        "outcomes_recorded": with_outcome,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "realised_cost": (
            false_negatives * COST_FALSE_NEGATIVE + false_positives * COST_FALSE_POSITIVE
        ),
    }
