"""What a fitted artifact needs before the platform's gate can judge it.

A score is comparable to another only when both were measured on the same held-out
data, so the record carries the digest of that data beside the figure. The split is
deterministic, which is what lets a digest identify it at all.

The metric is the expected loss under the dataset's own cost matrix, where a missed
default costs five times a rejected good applicant. Accuracy and ROC-AUC are
reported alongside it because they are how the model is usually described, and
neither is what the decision is worth.

Usage: python -m model.release [--publish]
"""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd
from registry.environment import capture
from registry.gate import digest_bytes
from registry.models import get
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from model.artifact import load
from model.config import (
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
    DECISION_THRESHOLD,
    MODEL_DIR,
    MODEL_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from model.preprocessing import load_data

log = logging.getLogger(__name__)

MODEL_NAME = "credit-risk"
RELEASE_PATH = MODEL_DIR / "release.json"
RELEASE_FILE = "release.json"
ARTIFACT_FILE = "credit_risk_model.joblib"


def holdout(source: str | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """The same split the training uses, reproduced rather than stored."""
    frame = load_data(source) if source else load_data()
    features = frame.drop(TARGET_COLUMN, axis=1)
    target = frame[TARGET_COLUMN]

    _, x_test, _, y_test = train_test_split(
        features, target, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=target
    )
    return x_test, y_test


def holdout_digest(x_test: pd.DataFrame, y_test: pd.Series) -> str:
    """Identifies the held-out rows, so a changed split reports as changed data."""
    frame = x_test.copy()
    frame[TARGET_COLUMN] = y_test
    return digest_bytes(frame.sort_index().to_csv(index=True).encode("utf-8"))


def total_cost(y_true, predictions) -> int:
    """Expected loss under the dataset's own cost matrix."""
    _, false_positive, false_negative, _ = confusion_matrix(y_true, predictions).ravel()
    return int(false_negative * COST_FALSE_NEGATIVE + false_positive * COST_FALSE_POSITIVE)


def measure(pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """The figure the gate turns on, and the ones a reader expects beside it."""
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= DECISION_THRESHOLD).astype(int)

    return {
        "metric": "total_cost",
        "value": total_cost(y_test, predictions),
        "rows": int(len(y_test)),
        "threshold": DECISION_THRESHOLD,
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
    }


def build_record(source: str | None = None) -> dict:
    """Measures the stored artifact on the held-out rows and describes the result."""
    artifact = load(MODEL_PATH)
    x_test, y_test = holdout(source)

    record = measure(artifact.pipeline, x_test, y_test)
    record["dataset"] = holdout_digest(x_test, y_test)
    record["artifact_sha256"] = artifact.version
    record["environment"] = capture(get(MODEL_NAME).packages)
    return record


CARD = """---
license: mit
library_name: sklearn
pipeline_tag: tabular-classification
tags:
  - credit-scoring
  - tabular
---

# Credit risk scorer

Scores an applicant for the German Credit dataset, where the positive class is a
bad credit risk. Promoted on expected loss rather than on accuracy.

## Metrics

Measured on {rows} held-out applicants at a decision threshold of {threshold}.

| Metric | Value |
|---|---:|
| Expected loss under the cost matrix | {value} |
| ROC-AUC | {roc_auc} |
| Accuracy | {accuracy} |

The cost matrix is the dataset's own: a missed default costs {fn} times a rejected
good applicant. Accuracy is reported because it is how a classifier is usually
described, and on this problem it is the least useful of the three: the threshold
that maximises it is not the one that minimises loss.

## Provenance

Held-out data digest `{dataset}`, which the platform's gate compares before it
compares any score. A candidate measured on other data is refused as unjudged
rather than promoted.

The artifact is a pickle and executes on load. The library versions it was written
under are recorded in `{release_file}`, and the platform refuses to serve it under
versions that could reconstruct the estimator differently.

Trained by [Credit-Risk-Assessment](https://github.com/jorgeasmz/Credit-Risk-Assessment)
and operated by [ML-Platform](https://github.com/jorgeasmz/ML-Platform).
"""


def build_card(record: dict) -> str:
    return CARD.format(release_file=RELEASE_FILE, fn=COST_FALSE_NEGATIVE, **{
        key: value for key, value in record.items()
        if key not in {"environment", "artifact_sha256", "metric", "dataset"}
    }, dataset=record["dataset"])


def publish(record: dict, repo: str) -> str:
    """Uploads the artifact, its record and its card. Returns the commit."""
    from huggingface_hub import HfApi

    card = MODEL_DIR / "README.md"
    card.write_text(build_card(record))

    api = HfApi()
    api.create_repo(repo, repo_type="model", exist_ok=True)
    commit = None
    for path, name in (
        (MODEL_PATH, ARTIFACT_FILE),
        (RELEASE_PATH, RELEASE_FILE),
        (card, "README.md"),
    ):
        info = api.upload_file(
            path_or_fileobj=str(path), path_in_repo=name, repo_id=repo, repo_type="model"
        )
        commit = getattr(info, "oid", None) or commit

    return commit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true", help="upload to the model repository")
    parser.add_argument("--repo", default=get(MODEL_NAME).repo)
    arguments = parser.parse_args()

    record = build_record()
    RELEASE_PATH.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

    print(f"{record['metric']}: {record['value']}  on {record['rows']} rows")
    print(f"held-out digest: {record['dataset']}")
    print(f"written to {RELEASE_PATH}")

    if arguments.publish:
        commit = publish(record, arguments.repo)
        print(f"published to https://huggingface.co/{arguments.repo} at {commit}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
