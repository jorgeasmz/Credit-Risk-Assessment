import json

import pytest

from model import fetch
from model.fetch import (
    PinMissing,
    VersionMismatch,
    blocking_differences,
    download,
    read_pin,
)

POINTER = {
    "model": "credit-risk",
    "version": "3",
    "repo": "jorgeasmz/credit-risk-scorer",
    "revision": "a" * 40,
    "artifact_file": "credit_risk_model.joblib",
    "url": "https://huggingface.co/jorgeasmz/credit-risk-scorer/resolve/"
           + "a" * 40 + "/credit_risk_model.joblib",
    "metric": "total_cost",
    "value": 96,
    "dataset": "ba349129c0b0c828",
    "environment": {
        "python": "3.12.3",
        "packages": {"scikit-learn": "1.9.0", "numpy": "2.5.2"},
    },
}


@pytest.fixture
def pin_file(tmp_path):
    path = tmp_path / "production.json"
    path.write_text(json.dumps(POINTER))
    return path


def present(monkeypatch, versions: dict[str, str]) -> None:
    monkeypatch.setattr(fetch, "installed", lambda name: versions.get(name, "absent"))


def test_a_missing_pin_says_a_promotion_writes_it(tmp_path):
    with pytest.raises(PinMissing, match="pass the gate|passes the gate"):
        read_pin(tmp_path / "nothing.json")


def test_the_pin_round_trips(pin_file):
    assert read_pin(pin_file)["revision"] == "a" * 40


def test_matching_versions_block_nothing(monkeypatch):
    present(monkeypatch, {"scikit-learn": "1.9.0", "numpy": "2.5.2"})

    assert blocking_differences(POINTER) == []


def test_a_patch_difference_does_not_block(monkeypatch):
    present(monkeypatch, {"scikit-learn": "1.9.4", "numpy": "2.5.2"})

    assert blocking_differences(POINTER) == []


def test_a_minor_difference_blocks(monkeypatch):
    """The failure this exists for: 1.8 reading what 1.9 wrote, without an exception."""
    present(monkeypatch, {"scikit-learn": "1.8.2", "numpy": "2.5.2"})

    assert blocking_differences(POINTER) == [
        "scikit-learn: recorded 1.9.0, present 1.8.2"
    ]


def test_an_absent_package_blocks(monkeypatch):
    present(monkeypatch, {"scikit-learn": "1.9.0"})

    assert blocking_differences(POINTER) == ["numpy: recorded 2.5.2, present absent"]


def test_a_pointer_without_an_environment_blocks_nothing(monkeypatch):
    present(monkeypatch, {})

    assert blocking_differences({"model": "credit-risk"}) == []


def test_the_build_refuses_a_mismatched_pin(monkeypatch, pin_file):
    present(monkeypatch, {"scikit-learn": "1.8.2", "numpy": "2.5.2"})
    monkeypatch.setattr(fetch, "PIN_PATH", pin_file)
    monkeypatch.setattr(fetch, "download", lambda *a, **k: pytest.fail("downloaded"))

    with pytest.raises(VersionMismatch, match="scikit-learn"):
        fetch.main()


def test_the_download_writes_the_body_to_the_artifact_path(monkeypatch, tmp_path):
    class Response:
        content = b"fitted"

        def raise_for_status(self):
            return None

    asked = {}

    def fake_get(url, timeout):
        asked.update(url=url, timeout=timeout)
        return Response()

    monkeypatch.setattr(fetch.requests, "get", fake_get)
    destination = tmp_path / "nested" / "model.joblib"

    written = download(POINTER, destination)

    assert written.read_bytes() == b"fitted"
    assert asked["url"] == POINTER["url"]


def test_a_pointer_without_a_url_is_refused():
    with pytest.raises(ValueError, match="no artifact URL"):
        download({"model": "credit-risk"})


def test_the_url_addresses_a_commit_rather_than_a_branch():
    """Pinned bytes: a branch moves, a commit does not."""
    assert f"/resolve/{POINTER['revision']}/" in POINTER["url"]
