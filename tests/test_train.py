import joblib

from model import train as train_module


def test_training_writes_a_usable_artifact(monkeypatch, tmp_path, training_frame):
    """Runs the whole training path offline, against the synthetic frame."""
    destination = tmp_path / "credit_risk_model.joblib"
    monkeypatch.setattr(train_module, "load_data", lambda: training_frame)
    monkeypatch.setattr(train_module, "MODEL_PATH", destination)

    train_module.train_model()

    assert destination.exists()
    pipeline = joblib.load(destination)
    assert pipeline.predict_proba(training_frame.head(1).drop("status", axis=1)).shape == (1, 2)
