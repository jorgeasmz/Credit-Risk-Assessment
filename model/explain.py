import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from model.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES


def _dense(matrix) -> np.ndarray:
    """OneHotEncoder may hand back a sparse matrix; SHAP wants an array."""
    return np.asarray(matrix.todense()) if hasattr(matrix, "todense") else np.asarray(matrix)


def transformed_feature_owners(pipeline: Pipeline) -> list[str]:
    """
    Maps each transformed column back to the field the applicant filled in.

    Built from the encoder's own fitted categories rather than by parsing
    get_feature_names_out(): names like "cat__checkin_acc_A11" cannot be split
    reliably when the column name itself contains underscores.
    """
    onehot = (
        pipeline.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .named_steps["onehot"]
    )

    owners = list(NUMERICAL_FEATURES)
    for column, categories in zip(CATEGORICAL_FEATURES, onehot.categories_, strict=True):
        owners.extend([column] * len(categories))
    return owners


def build_background(pipeline: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    """Transformed training rows, used as the reference distribution."""
    return _dense(pipeline.named_steps["preprocessor"].transform(frame))


def build_explainer(pipeline: Pipeline, background: np.ndarray) -> shap.LinearExplainer:
    """
    Builds the explainer once, over the whole background.

    SHAP subsamples the background by default, which makes the same application
    explainable two different ways on two different days. For a decision that has
    to be defensible after the fact, the explanation must be reproducible, so the
    masker is pinned to every row instead.
    """
    masker = shap.maskers.Independent(background, max_samples=len(background))
    return shap.LinearExplainer(pipeline.named_steps["classifier"], masker)


def explain(
    pipeline: Pipeline, explainer: shap.LinearExplainer, frame: pd.DataFrame
) -> dict:
    """
    Per-field contribution to the log-odds of default for a single applicant.

    SHAP runs on the transformed matrix, where one categorical field occupies a
    column per level. Summing those back onto the original field is what turns
    sixty unreadable numbers into "the checking account balance moved this
    decision", which is the form a declined applicant is entitled to.

    Positive values push toward default, negative values away from it.
    """
    transformed = _dense(pipeline.named_steps["preprocessor"].transform(frame))
    values = np.asarray(explainer.shap_values(transformed)).reshape(-1)

    contributions: dict[str, float] = {}
    for owner, value in zip(transformed_feature_owners(pipeline), values, strict=True):
        contributions[owner] = contributions.get(owner, 0.0) + float(value)

    # Largest absolute effect first: that is the order an analyst reads them in.
    return dict(sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True))
