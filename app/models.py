from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class Decision(Base):
    """
    One scoring decision, kept for audit.

    A credit decision has to be reconstructable after the fact: what was asked,
    what the model answered, which artifact answered it and under which
    threshold. Storing the probability alone would not survive a dispute.
    """

    __tablename__ = "decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True, nullable=False
    )

    # Which artifact produced this, and under which rule.
    model_version: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)

    risk_class: Mapped[int] = mapped_column(Integer, nullable=False)
    probability: Mapped[float] = mapped_column(Float, nullable=False)

    application: Mapped[dict] = mapped_column(JSON, nullable=False)
    contributions: Mapped[dict] = mapped_column(JSON, nullable=False)

    # Feedback loop. Cost cannot be measured without knowing what actually
    # happened, so the ground truth arrives later or not at all.
    defaulted: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    outcome_recorded_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
