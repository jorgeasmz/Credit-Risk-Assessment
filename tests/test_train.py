from model import train as train_module
from model.artifact import load


def test_training_writes_a_usable_artifact(monkeypatch, tmp_path, training_frame):
    """Runs the whole training path offline, against the synthetic frame."""
    destination = tmp_path / "credit_risk_model.joblib"
    monkeypatch.setattr(train_module, "load_data", lambda: training_frame)
    monkeypatch.setattr(train_module, "MODEL_PATH", destination)

    train_module.train_model()

    assert destination.exists()

    artifact = load(destination)
    features = training_frame.head(1).drop("status", axis=1)
    assert artifact.pipeline.predict_proba(features).shape == (1, 2)


def test_the_artifact_carries_its_explanation_background(
    monkeypatch, tmp_path, training_frame
):
    """
    The background is fitted at training time and shipped with the model, so the
    service never needs the training set to explain a prediction.
    """
    destination = tmp_path / "credit_risk_model.joblib"
    monkeypatch.setattr(train_module, "load_data", lambda: training_frame)
    monkeypatch.setattr(train_module, "MODEL_PATH", destination)

    train_module.train_model()

    artifact = load(destination)
    assert artifact.background.shape[0] > 0
    assert artifact.version
