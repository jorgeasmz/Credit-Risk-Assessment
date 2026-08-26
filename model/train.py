"""
Training entry point: downloads the dataset, fits the pipeline and writes the
artifact next to this file.

Usage: python -m model.train
"""

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from model.artifact import save
from model.config import (
    DECISION_THRESHOLD,
    MODEL_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from model.explain import build_background
from model.pipeline import build_pipeline
from model.preprocessing import load_data


def train_model() -> None:
    print("Loading data...")
    df = load_data()

    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

    print(f"Rows: {len(X)}  |  Positive class (bad credit): {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print("Training pipeline...")
    clf = build_pipeline()
    clf.fit(X_train, y_train)

    probabilities = clf.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= DECISION_THRESHOLD).astype(int)
    majority_baseline = max(y_test.mean(), 1 - y_test.mean())

    print(f"\nThreshold:         {DECISION_THRESHOLD}")
    print(f"Accuracy:          {accuracy_score(y_test, predictions):.4f}")
    print(f"Majority baseline: {majority_baseline:.4f}")
    print(f"ROC-AUC:           {roc_auc_score(y_test, probabilities):.4f}\n")
    print(classification_report(y_test, predictions, digits=3))
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(y_test, predictions))

    # The background travels with the model; the service never sees the training set.
    save(clf, build_background(clf, X_train), MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
