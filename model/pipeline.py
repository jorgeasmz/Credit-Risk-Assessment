from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from model.config import (
    CATEGORICAL_FEATURES,
    N_ESTIMATORS,
    NUMERICAL_FEATURES,
    RANDOM_STATE,
)


def build_classifier():
    """
    The production classifier.

    Logistic regression rather than the forest: on this dataset it costs less
    under the UCI cost matrix, ranks better (ROC-AUC 0.806 vs 0.788) and, unlike
    the forest, exposes coefficients that can justify a declined application.
    See evaluate.py for the comparison.
    """
    return LogisticRegression(
        max_iter=1000,
        # The target is 70/30; without this the model under-predicts risk.
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def build_forest():
    """The Random Forest baseline the production choice is measured against."""
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def build_pipeline(classifier=None) -> Pipeline:
    """
    Preprocessing and classifier as a single estimator.

    Keeping the transformers inside the pipeline is what makes training and
    serving impossible to desynchronise: the API hands over raw fields and the
    pipeline applies the exact transformations it was fitted with.
    """
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_transformer, NUMERICAL_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier if classifier is not None else build_classifier()),
    ])
