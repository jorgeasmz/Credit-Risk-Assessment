import joblib
import pytest

from app.utils import format_prediction, load_model


def test_load_model_reports_a_missing_artifact(tmp_path):
    with pytest.raises(FileNotFoundError, match="model.train"):
        load_model(tmp_path / "absent.joblib")


def test_load_model_round_trips(tmp_path, fitted_pipeline):
    path = tmp_path / "model.joblib"
    joblib.dump(fitted_pipeline, path)

    assert load_model(path) is not None


def test_format_prediction_shapes_the_response(fitted_pipeline, valid_payload):
    result = format_prediction(fitted_pipeline, valid_payload)

    assert set(result) == {"risk_class", "risk_label", "probability"}
    assert result["risk_class"] in (0, 1)
    assert 0.0 <= result["probability"] <= 1.0


def test_label_follows_the_class(fitted_pipeline, valid_payload):
    result = format_prediction(fitted_pipeline, valid_payload)

    expected = "High Risk" if result["risk_class"] == 1 else "Low Risk"
    assert result["risk_label"] == expected


@pytest.mark.parametrize(
    ("threshold", "expected"), [(0.0, 1), (1.01, 0)]
)
def test_threshold_governs_the_class(
    fitted_pipeline, valid_payload, threshold, expected
):
    """The class comes from the threshold, not from predict()'s implicit 0.5."""
    result = format_prediction(fitted_pipeline, valid_payload, threshold=threshold)

    assert result["risk_class"] == expected
