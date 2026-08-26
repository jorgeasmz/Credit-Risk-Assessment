import pytest
from fastapi.testclient import TestClient

from app import settings
from app.main import app, get_scorer


def test_health_reports_which_model_is_loaded(client):
    body = client.get("/").json()

    assert body["model_loaded"] is True
    assert body["model_version"] == "testartifact"


def test_health_needs_no_credentials(client):
    assert client.get("/").status_code == 200


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("post", "/predict"),
        ("get", "/decisions"),
        ("get", "/decisions/1"),
        ("post", "/decisions/1/outcome"),
        ("get", "/summary"),
    ],
)
def test_everything_that_touches_an_applicant_requires_a_key(client, method, path):
    body = {"json": {}} if method == "post" else {}

    response = getattr(client, method)(path, **body)

    assert response.status_code == 401


def test_a_wrong_key_is_rejected(client, valid_payload):
    response = client.post(
        "/predict", json=valid_payload, headers={settings.API_KEY_HEADER: "wrong"}
    )

    assert response.status_code == 401


def test_an_unconfigured_server_fails_closed(client, valid_payload, auth, monkeypatch):
    """No key configured must mean no scoring, not scoring for everyone."""
    monkeypatch.setattr(settings, "API_KEY", "")

    response = client.post("/predict", json=valid_payload, headers=auth)

    assert response.status_code == 503


def test_predict_returns_the_decision_and_its_explanation(client, valid_payload, auth):
    body = client.post("/predict", json=valid_payload, headers=auth).json()

    assert body["decision_id"] > 0
    assert body["risk_class"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert body["model_version"] == "testartifact"
    assert body["contributions"]


def test_predict_writes_the_decision_to_the_log(client, valid_payload, auth):
    created = client.post("/predict", json=valid_payload, headers=auth).json()

    stored = client.get(f"/decisions/{created['decision_id']}", headers=auth).json()

    assert stored["application"]["amount"] == valid_payload["amount"]
    assert stored["probability"] == pytest.approx(created["probability"])
    assert stored["defaulted"] is None


def test_predict_rejects_an_underage_applicant(client, valid_payload, auth):
    response = client.post(
        "/predict", json={**valid_payload, "age": 17}, headers=auth
    )

    assert response.status_code == 422


def test_predict_rejects_a_missing_field(client, valid_payload, auth):
    del valid_payload["amount"]

    assert client.post("/predict", json=valid_payload, headers=auth).status_code == 422


def test_predict_does_not_leak_internal_errors(client, valid_payload, auth):
    class Exploding:
        version = "boom"

        def score(self, *args, **kwargs):
            raise RuntimeError("secret internal detail")

    app.dependency_overrides[get_scorer] = lambda: Exploding()
    try:
        response = client.post("/predict", json=valid_payload, headers=auth)
    finally:
        app.dependency_overrides.pop(get_scorer, None)

    assert response.status_code == 500
    assert "secret internal detail" not in response.text


def test_the_log_is_paginated_newest_first(client, valid_payload, auth):
    ids = [
        client.post("/predict", json=valid_payload, headers=auth).json()["decision_id"]
        for _ in range(5)
    ]

    first = client.get("/decisions?limit=2", headers=auth).json()
    second = client.get(
        f"/decisions?limit=2&cursor={first['next_cursor']}", headers=auth
    ).json()

    assert [item["id"] for item in first["items"]] == ids[-1:-3:-1]
    assert [item["id"] for item in second["items"]] == ids[-3:-5:-1]


def test_the_page_size_is_bounded(client, auth):
    response = client.get(f"/decisions?limit={settings.MAX_PAGE_SIZE + 1}", headers=auth)

    assert response.status_code == 422


def test_an_unknown_decision_is_a_404(client, auth):
    assert client.get("/decisions/9999", headers=auth).status_code == 404
    assert (
        client.post(
            "/decisions/9999/outcome", json={"defaulted": True}, headers=auth
        ).status_code
        == 404
    )


def test_recording_an_outcome_closes_the_loop(client, valid_payload, auth):
    decision_id = client.post("/predict", json=valid_payload, headers=auth).json()[
        "decision_id"
    ]

    updated = client.post(
        f"/decisions/{decision_id}/outcome", json={"defaulted": True}, headers=auth
    ).json()

    assert updated["defaulted"] == 1
    assert updated["outcome_recorded_at"] is not None


def test_summary_prices_only_what_it_can_observe(client, valid_payload, auth):
    decision = client.post("/predict", json=valid_payload, headers=auth).json()

    before = client.get("/summary", headers=auth).json()
    client.post(
        f"/decisions/{decision['decision_id']}/outcome",
        json={"defaulted": True},
        headers=auth,
    )
    after = client.get("/summary", headers=auth).json()

    assert before["outcomes_recorded"] == 0
    assert before["realised_cost"] == 0
    assert after["outcomes_recorded"] == 1
    assert after["total"] == 1


def test_the_example_reaches_the_openapi_document(client):
    schema = client.get("/openapi.json").json()

    assert "example" in schema["components"]["schemas"]["CreditApplication"]


def test_the_key_header_is_documented(client):
    """A consumer should learn how to authenticate from the schema alone."""
    schema = client.get("/openapi.json").json()

    schemes = schema["components"]["securitySchemes"]
    assert any(s.get("name") == settings.API_KEY_HEADER for s in schemes.values())


def test_lifespan_survives_a_missing_artifact(monkeypatch):
    """The service must start and explain itself, not crash-loop."""

    def explode(*args, **kwargs):
        raise FileNotFoundError("no artifact")

    monkeypatch.setattr("app.main.load_scorer", explode)

    with TestClient(app) as client:
        body = client.get("/").json()

    assert body["model_loaded"] is False
    assert body["model_version"] is None


def test_summary_exposes_the_score_distribution(client, valid_payload, auth):
    client.post("/predict", json=valid_payload, headers=auth)

    summary = client.get("/summary", headers=auth).json()

    assert len(summary["score_distribution"]) == settings.SCORE_BUCKETS
    assert sum(summary["score_distribution"]) == summary["total"]
