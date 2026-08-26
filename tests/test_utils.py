import pytest


def test_score_returns_the_documented_shape(fitted_scorer, valid_payload):
    result = fitted_scorer.score(valid_payload)

    assert set(result) == {
        "risk_class",
        "risk_label",
        "probability",
        "threshold",
        "model_version",
        "contributions",
    }
    assert 0.0 <= result["probability"] <= 1.0


def test_label_follows_the_class(fitted_scorer, valid_payload):
    result = fitted_scorer.score(valid_payload)

    expected = "High Risk" if result["risk_class"] == 1 else "Low Risk"
    assert result["risk_label"] == expected


@pytest.mark.parametrize(("threshold", "expected"), [(0.0, 1), (1.01, 0)])
def test_threshold_governs_the_class(fitted_scorer, valid_payload, threshold, expected):
    """The class comes from the threshold, not from predict()'s implicit 0.5."""
    result = fitted_scorer.score(valid_payload, threshold=threshold)

    assert result["risk_class"] == expected
    assert result["threshold"] == threshold


def test_every_decision_carries_an_explanation(fitted_scorer, valid_payload):
    result = fitted_scorer.score(valid_payload)

    assert result["contributions"]
    assert all(isinstance(v, float) for v in result["contributions"].values())


def test_the_artifact_version_travels_with_the_score(fitted_scorer, valid_payload):
    assert fitted_scorer.score(valid_payload)["model_version"] == fitted_scorer.version
