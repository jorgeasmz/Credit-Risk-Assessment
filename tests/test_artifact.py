import numpy as np
import pytest

from model.artifact import content_hash, load, save
from model.config import TARGET_COLUMN
from model.explain import build_background


def test_round_trips_the_pipeline_and_its_background(
    tmp_path, fitted_pipeline, training_frame
):
    path = tmp_path / "model.joblib"
    background = build_background(
        fitted_pipeline, training_frame.drop(TARGET_COLUMN, axis=1)
    )

    save(fitted_pipeline, background, path)
    artifact = load(path)

    assert artifact.pipeline.predict_proba(
        training_frame.drop(TARGET_COLUMN, axis=1).head(1)
    ).shape == (1, 2)
    assert np.array_equal(artifact.background, background)


def test_save_creates_missing_directories(tmp_path, fitted_pipeline, training_frame):
    destination = tmp_path / "nested" / "model.joblib"
    background = build_background(
        fitted_pipeline, training_frame.drop(TARGET_COLUMN, axis=1)
    )

    save(fitted_pipeline, background, destination)

    assert destination.exists()


def test_version_identifies_the_bytes_not_a_label(tmp_path):
    """Two builds of the same version number are not the same model."""
    first = tmp_path / "a.bin"
    second = tmp_path / "b.bin"
    first.write_bytes(b"identical")
    second.write_bytes(b"identical")
    different = tmp_path / "c.bin"
    different.write_bytes(b"something else")

    assert content_hash(first) == content_hash(second)
    assert content_hash(first) != content_hash(different)
    assert len(content_hash(first)) == 12


def test_loading_a_missing_artifact_says_how_to_build_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="model.train"):
        load(tmp_path / "absent.joblib")
