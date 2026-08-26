from app import repository
from app.models import Decision


def _decision(session, **overrides):
    defaults = {
        "model_version": "abc123",
        "threshold": 0.45,
        "risk_class": 0,
        "probability": 0.2,
        "application": {"amount": 1000},
        "contributions": {"amount": -0.1},
    }
    return repository.record_decision(session, **{**defaults, **overrides})


def test_records_a_decision_with_an_id(session):
    decision = _decision(session)

    assert decision.id is not None
    assert decision.created_at is not None
    assert session.get(Decision, decision.id) is not None


def test_lists_newest_first(session):
    for probability in (0.1, 0.2, 0.3):
        _decision(session, probability=probability)

    page, _ = repository.list_decisions(session, limit=10)

    assert [d.probability for d in page] == [0.3, 0.2, 0.1]


def test_pagination_neither_skips_nor_repeats(session):
    for _ in range(5):
        _decision(session)

    first, cursor = repository.list_decisions(session, limit=2)
    second, next_cursor = repository.list_decisions(session, limit=2, cursor=cursor)
    third, last_cursor = repository.list_decisions(session, limit=2, cursor=next_cursor)

    seen = [d.id for d in first + second + third]
    assert seen == sorted(seen, reverse=True)
    assert len(set(seen)) == 5
    assert last_cursor is None


def test_cursor_is_absent_on_the_final_page(session):
    _decision(session)

    _, cursor = repository.list_decisions(session, limit=10)

    assert cursor is None


def test_records_an_outcome(session):
    decision = _decision(session)

    updated = repository.record_outcome(session, decision.id, defaulted=True)

    assert updated is not None
    assert updated.defaulted == 1
    assert updated.outcome_recorded_at is not None


def test_recording_an_outcome_for_an_unknown_decision_returns_none(session):
    assert repository.record_outcome(session, 999, defaulted=False) is None


def test_summary_of_an_empty_log_does_not_divide_by_zero(session):
    summary = repository.summarise(session)

    assert summary["total"] == 0
    assert summary["approval_rate"] == 0.0
    assert summary["realised_cost"] == 0


def test_summary_prices_mistakes_with_the_cost_matrix(session):
    approved_and_defaulted = _decision(session, risk_class=0)
    rejected_and_repaid = _decision(session, risk_class=1)
    approved_and_repaid = _decision(session, risk_class=0)

    repository.record_outcome(session, approved_and_defaulted.id, defaulted=True)
    repository.record_outcome(session, rejected_and_repaid.id, defaulted=False)
    repository.record_outcome(session, approved_and_repaid.id, defaulted=False)

    summary = repository.summarise(session)

    assert summary["false_negatives"] == 1
    assert summary["false_positives"] == 1
    # A missed default costs five times a rejected good applicant.
    assert summary["realised_cost"] == 5 + 1


def test_summary_ignores_decisions_without_an_outcome(session):
    _decision(session, risk_class=0)

    summary = repository.summarise(session)

    assert summary["total"] == 1
    assert summary["outcomes_recorded"] == 0
    assert summary["realised_cost"] == 0
