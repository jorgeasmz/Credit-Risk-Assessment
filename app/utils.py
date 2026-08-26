import logging

import pandas as pd

from model.artifact import ModelArtifact, load
from model.config import DECISION_THRESHOLD, MODEL_PATH
from model.explain import build_explainer, explain

logger = logging.getLogger(__name__)


class Scorer:
    """A loaded artifact and its explainer, held for the life of the process."""

    def __init__(self, artifact: ModelArtifact):
        self.artifact = artifact
        self.explainer = build_explainer(artifact.pipeline, artifact.background)

    @property
    def version(self) -> str:
        return self.artifact.version

    def score(self, application: dict, threshold: float = DECISION_THRESHOLD) -> dict:
        """Scores one application against an explicit threshold and explains it."""
        frame = pd.DataFrame([application])

        probability = float(self.artifact.pipeline.predict_proba(frame)[0][1])
        risk_class = int(probability >= threshold)

        return {
            "risk_class": risk_class,
            "risk_label": "High Risk" if risk_class else "Low Risk",
            "probability": probability,
            "threshold": threshold,
            "model_version": self.version,
            "contributions": explain(self.artifact.pipeline, self.explainer, frame),
        }


def load_scorer(path=MODEL_PATH) -> Scorer:
    """Loads the artifact from disk and prepares it for serving."""
    logger.info("Loading model from %s", path)
    scorer = Scorer(load(path))
    logger.info("Model %s ready.", scorer.version)
    return scorer
