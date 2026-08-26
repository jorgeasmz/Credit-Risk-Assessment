"""
Evaluates the credit risk pipeline against baselines and picks the decision
threshold that minimises expected cost.

The UCI German Credit dataset ships a cost matrix: approving an applicant who
defaults costs 5, rejecting one who would have repaid costs 1. Optimising for
accuracy on this data means ignoring that asymmetry, which is why the report
leads with cost.

Usage: python -m evaluate
"""

from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from model.config import (
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from model.pipeline import build_forest, build_pipeline
from model.preprocessing import load_data

THRESHOLDS = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)


def total_cost(y_true, predictions) -> int:
    """Expected loss under the dataset's own cost matrix."""
    _, false_pos, false_neg, _ = confusion_matrix(y_true, predictions).ravel()
    return false_neg * COST_FALSE_NEGATIVE + false_pos * COST_FALSE_POSITIVE


def main() -> None:
    df = load_data()
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")
    print(f"Bad credit in test: {y_test.mean():.2%}\n")

    models = {
        "Logistic Regression": build_pipeline(),
        "Random Forest": build_pipeline(build_forest()),
        "Majority class": DummyClassifier(strategy="most_frequent"),
    }

    header = (
        f"{'Model':<22}{'Acc':>7}{'ROC-AUC':>10}{'PR-AUC':>9}"
        f"{'Prec':>7}{'Recall':>8}{'F1':>7}{'Cost':>8}"
    )
    print(header)
    print("-" * len(header))

    production = None
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        if name == "Logistic Regression":
            production = model
        print(
            f"{name:<22}{accuracy_score(y_test, predictions):>7.3f}"
            f"{roc_auc_score(y_test, probabilities):>10.3f}"
            f"{average_precision_score(y_test, probabilities):>9.3f}"
            f"{precision_score(y_test, predictions, zero_division=0):>7.3f}"
            f"{recall_score(y_test, predictions, zero_division=0):>8.3f}"
            f"{f1_score(y_test, predictions, zero_division=0):>7.3f}"
            f"{total_cost(y_test, predictions):>8}"
        )

    # Two trivial policies bound the problem: a model that costs more than
    # rejecting everybody is worse than having no model at all.
    approve_all = total_cost(y_test, [0] * len(y_test))
    reject_all = total_cost(y_test, [1] * len(y_test))
    print(f"\nApprove everyone: {approve_all}   Reject everyone: {reject_all}")

    probabilities = production.predict_proba(X_test)[:, 1]
    print("\nProduction model, threshold sweep:")
    sweep = f"{'Threshold':<12}{'Precision':>11}{'Recall':>9}{'Accuracy':>10}{'Cost':>8}"
    print(sweep)
    print("-" * len(sweep))

    best = (10**9, 0.5)
    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype(int)
        cost = total_cost(y_test, predictions)
        print(
            f"{threshold:<12.2f}"
            f"{precision_score(y_test, predictions, zero_division=0):>11.3f}"
            f"{recall_score(y_test, predictions, zero_division=0):>9.3f}"
            f"{accuracy_score(y_test, predictions):>10.3f}{cost:>8}"
        )
        if cost < best[0]:
            best = (cost, threshold)

    print(f"\nLowest cost: {best[0]} at threshold {best[1]:.2f}")


if __name__ == "__main__":
    main()
