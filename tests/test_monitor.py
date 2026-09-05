import pytest

from app.models import Decision
from model import monitor
from model.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES


def log(session, application: dict, probability: float, count: int = 1) -> None:
    session.add_all(
        Decision(
            model_version="testartifact", threshold=0.45, risk_class=0,
            probability=probability, application=dict(application), contributions={},
        )
        for _ in range(count)
    )
    session.commit()


def test_an_empty_log_reads_as_no_columns(session):
    observed = monitor.current(session)

    assert observed[monitor.SCORE] == []
    assert monitor.insufficient(observed, monitor.MINIMUM) is True


def test_the_log_is_read_as_columns_in_the_reference_shape(session, valid_payload):
    log(session, valid_payload, 0.3, count=3)

    observed = monitor.current(session)

    assert set(observed) == set(monitor.FEATURES) | {monitor.SCORE}
    assert observed[monitor.SCORE] == [0.3, 0.3, 0.3]
    assert observed["amount"] == [valid_payload["amount"]] * 3


def test_a_short_log_is_insufficient(session, valid_payload):
    log(session, valid_payload, 0.3, count=monitor.MINIMUM - 1)

    assert monitor.insufficient(monitor.current(session), monitor.MINIMUM) is True


def test_a_long_enough_log_is_sufficient(session, valid_payload):
    log(session, valid_payload, 0.3, count=monitor.MINIMUM)

    assert monitor.insufficient(monitor.current(session), monitor.MINIMUM) is False


def test_a_field_missing_from_a_stored_application_does_not_shift_the_others(
    session, valid_payload
):
    """A record written by an older schema must not misalign the columns."""
    partial = {k: v for k, v in valid_payload.items() if k != "telephone"}
    log(session, valid_payload, 0.3)
    log(session, partial, 0.4)

    observed = monitor.current(session)

    assert observed["telephone"] == [valid_payload["telephone"]]
    assert observed["amount"] == [valid_payload["amount"]] * 2
    assert observed[monitor.SCORE] == [0.3, 0.4]


def test_the_score_is_compared_as_well_as_the_features():
    """Features can each stay put while the model's output moves."""
    assert monitor.SCORE not in monitor.FEATURES
    assert monitor.SCORE == "probability"


def test_every_declared_feature_is_one_the_model_uses():
    assert set(monitor.FEATURES) == set(CATEGORICAL_FEATURES) | set(NUMERICAL_FEATURES)


def test_the_report_names_every_column_and_its_severity():
    from registry.drift import Signal

    body = monitor.report([Signal("amount", 0.312, False), Signal("age", 0.02, False)], 250)

    assert "Compared 250 scored applications" in body
    assert "| `amount` | 0.312 | significant |" in body
    assert "| `age` | 0.020 | stable |" in body


def test_no_summary_is_written_outside_a_workflow(monkeypatch):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

    monitor.summarise("nothing should happen")


def test_the_summary_is_written_when_a_workflow_provides_one(monkeypatch, tmp_path):
    path = tmp_path / "summary.md"
    path.touch()
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(path))

    monitor.summarise("the body")

    assert "### Input drift" in path.read_text()
    assert "the body" in path.read_text()


@pytest.mark.parametrize("index,expected", [(0.05, 0), (0.30, 1)])
def test_exit_status_carries_the_alert(monkeypatch, session, valid_payload, index, expected):
    """A scheduled run that fails is the notification a free plan offers."""
    from registry.drift import Signal

    log(session, valid_payload, 0.3, count=monitor.MINIMUM)
    monkeypatch.setattr(monitor, "reference", lambda: {})
    monkeypatch.setattr(
        monitor, "compare", lambda reference, current, categorical: [
            Signal("amount", index, False)
        ]
    )

    class Factory:
        def __call__(self):
            return self

        def __enter__(self):
            return session

        def __exit__(self, *args):
            return False

    monkeypatch.setattr("app.db.SessionLocal", Factory())

    assert monitor.main() == expected
