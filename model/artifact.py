import hashlib
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class ModelArtifact:
    """A fitted pipeline plus what is needed to explain and identify it."""

    pipeline: Pipeline
    background: np.ndarray
    version: str


def content_hash(path: Path) -> str:
    """Short digest of the artifact bytes, recorded on every decision."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def save(pipeline: Pipeline, background: np.ndarray, path) -> None:
    """Writes the pipeline together with the background its explanations need."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "background": background}, path)


def load(path) -> ModelArtifact:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found at {path}. Run 'python -m model.train' first."
        )

    payload = joblib.load(path)
    return ModelArtifact(
        pipeline=payload["pipeline"],
        background=payload["background"],
        version=content_hash(path),
    )
