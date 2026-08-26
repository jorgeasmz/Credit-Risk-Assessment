import logging
from pathlib import Path

import joblib
import pandas as pd

from model.config import DECISION_THRESHOLD, MODEL_PATH

# A library module gets a logger; configuring handlers is the entry point's job.
logger = logging.getLogger(__name__)


def load_model(path=MODEL_PATH):
    """
    Loads the serialized scikit-learn pipeline.

    The default path is resolved from the model package, so the process working
    directory does not matter.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found at {path}. Run 'python -m model.train' first."
        )

    logger.info("Loading model from %s", path)
    return joblib.load(path)


def format_prediction(
    model, input_data: dict, threshold: float = DECISION_THRESHOLD
) -> dict:
    """
    Runs inference on one application and shapes the API response.

    The class comes from comparing the probability against an explicit
    threshold rather than from predict(), whose implicit 0.5 cut-off ignores
    that a missed default costs far more than a rejected good applicant.
    """
    frame = pd.DataFrame([input_data])

    probability = float(model.predict_proba(frame)[0][1])
    risk_class = int(probability >= threshold)

    return {
        "risk_class": risk_class,
        "risk_label": "High Risk" if risk_class else "Low Risk",
        "probability": probability,
    }
