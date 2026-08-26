import pytest
from fastapi.testclient import TestClient

from app.main import app, get_model


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def client_with_model(fitted_pipeline):
    app.dependency_overrides[get_model] = lambda: fitted_pipeline
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health_check_reports_model_availability(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is False


def test_predict_returns_the_documented_schema(client_with_model, valid_payload):
    response = client_with_model.post("/predict", json=valid_payload)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"risk_class", "risk_label", "probability"}


def test_predict_rejects_an_underage_applicant(client_with_model, valid_payload):
    response = client_with_model.post(
        "/predict", json={**valid_payload, "age": 17}
    )

    assert response.status_code == 422


def test_predict_rejects_a_missing_field(client_with_model, valid_payload):
    del valid_payload["amount"]

    response = client_with_model.post("/predict", json=valid_payload)

    assert response.status_code == 422


def test_predict_is_unavailable_without_a_model(client, valid_payload):
    """Startup leaves state.model None when the artifact is absent."""
    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 503


def test_predict_does_not_leak_internal_errors(valid_payload):
    """A failing model must not surface its exception text to the caller."""
    class Exploding:
        def predict_proba(self, frame):
            raise RuntimeError("secret internal detail")

    app.dependency_overrides[get_model] = lambda: Exploding()
    try:
        response = TestClient(app).post("/predict", json=valid_payload)
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 500
    assert "secret internal detail" not in response.text


def test_the_example_reaches_the_openapi_document(client):
    """The schema config typo used to drop it silently."""
    schema = client.get("/openapi.json").json()
    application = schema["components"]["schemas"]["CreditApplication"]

    assert "example" in application


def test_lifespan_publishes_the_model_on_startup(monkeypatch, fitted_pipeline):
    monkeypatch.setattr("app.main.load_model", lambda: fitted_pipeline)

    with TestClient(app) as client:
        assert client.get("/").json()["model_loaded"] is True


def test_lifespan_survives_a_missing_artifact(monkeypatch):
    """The service must still start and explain itself, not crash-loop."""
    def explode():
        raise FileNotFoundError("no artifact")

    monkeypatch.setattr("app.main.load_model", explode)

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is False
