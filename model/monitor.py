"""Whether the applications being scored still look like the ones fitted on.

The model was fitted on one population and is asked about whatever arrives. Nothing
in the serving path notices when those stop being the same, and the expected loss
that justified the promotion was measured on the first one.

The reference is the training split, reproduced from the same deterministic split
the fit used. The current sample is the decision log, which stores each application
as it was received. The score distribution is compared too: features can each stay
put while the model's output moves, and the output is what the decision is made on.

Exit status is the alert. A scheduled run that fails is a notification, which is
the mechanism a free plan offers; a run that reports drift in a summary nobody
opens is not one.

Usage: python -m model.monitor
"""

from __future__ import annotations

import os
import sys

from registry.drift import MODERATE, compare, insufficient
from sklearn.model_selection import train_test_split
from sqlalchemy import select

from model.artifact import load
from model.config import (
    CATEGORICAL_FEATURES,
    MODEL_PATH,
    NUMERICAL_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from model.preprocessing import load_data

# Below this the log is not a distribution, it is a handful of rows.
MINIMUM = 100

SCORE = "probability"
FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES


def reference() -> dict[str, list]:
    """The training split, and the scores the fitted model gives it."""
    frame = load_data()
    features = frame.drop(TARGET_COLUMN, axis=1)
    target = frame[TARGET_COLUMN]

    x_train, _, _, _ = train_test_split(
        features, target, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=target
    )

    columns = {name: x_train[name].tolist() for name in FEATURES}
    columns[SCORE] = load(MODEL_PATH).pipeline.predict_proba(x_train)[:, 1].tolist()
    return columns


def current(session) -> dict[str, list]:
    """The decision log, as columns, in the shape the reference is in."""
    from app.models import Decision

    rows = session.execute(select(Decision.application, Decision.probability)).all()
    columns: dict[str, list] = {name: [] for name in FEATURES}
    columns[SCORE] = []

    for application, probability in rows:
        for name in FEATURES:
            if name in application:
                columns[name].append(application[name])
        columns[SCORE].append(probability)
    return columns


def report(signals: list, scored: int) -> str:
    lines = [
        f"Compared {scored} scored applications against the training split.",
        "",
        "| Column | Index | |",
        "|---|---:|---|",
    ]
    lines += [
        f"| `{signal.column}` | {signal.index:.3f} | {signal.severity} |"
        for signal in signals
    ]
    return "\n".join(lines)


def main() -> int:
    from app.db import SessionLocal

    with SessionLocal() as session:
        observed = current(session)

    scored = len(observed.get(SCORE, []))
    if insufficient(observed, MINIMUM):
        message = (
            f"{scored} scored applications, fewer than the {MINIMUM} below which "
            "nothing is claimed. No comparison was made."
        )
        print(message)
        summarise(message)
        return 0

    signals = compare(reference(), observed, categorical=set(CATEGORICAL_FEATURES))
    moved = [signal for signal in signals if signal.index >= MODERATE]

    print(report(signals, scored))
    summarise(report(signals, scored))

    if moved:
        print(f"\n{len(moved)} column(s) moved significantly: "
              + ", ".join(signal.column for signal in moved))
        return 1
    return 0


def summarise(body: str) -> None:
    path = os.getenv("GITHUB_STEP_SUMMARY")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as summary:
        summary.write(f"### Input drift\n\n{body}\n\n")


if __name__ == "__main__":
    sys.exit(main())
