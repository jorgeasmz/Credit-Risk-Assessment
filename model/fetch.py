"""Downloads the model version the registry pinned, instead of fitting a new one.

A build that trains produces a model nothing measured and nothing compared against
what it replaces. This reads the pointer a promotion wrote and fetches exactly the
bytes that were scored, addressed by the commit they live at.

The pointer holds no credential and the repository is public, so the image needs
neither a token nor a route to the registry.

The version check below states the same rule the platform states, deliberately. The
serving image must be able to refuse a model it cannot read without the platform
installed in it, and a guard that depends on the thing it guards is not one.

Usage: python -m model.fetch
"""

from __future__ import annotations

import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import requests

from model.config import MODEL_DIR, MODEL_PATH

PIN_PATH = MODEL_DIR / "production.json"
TIMEOUT = 120
ABSENT = "absent"


class PinMissing(FileNotFoundError):
    """No version has been pinned, so there is nothing to build the image around."""


class VersionMismatch(RuntimeError):
    """The artifact was written under versions this image cannot be trusted to read."""


def read_pin(path: Path | None = None) -> dict:
    # Resolved on the call rather than bound at definition, so the location stays
    # one constant instead of a default that cannot be redirected.
    path = Path(path) if path is not None else PIN_PATH
    if not path.exists():
        raise PinMissing(
            f"No pinned version at {path}. A promotion writes it; until one passes "
            "the gate there is nothing to serve."
        )
    return json.loads(Path(path).read_text())


def installed(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return ABSENT


def blocking_differences(pointer: dict) -> list[str]:
    """Recorded versions that differ from the present ones by more than a patch."""
    recorded = pointer.get("environment", {}).get("packages", {})
    found = []
    for name, expected in sorted(recorded.items()):
        present = installed(name)
        if present == expected:
            continue
        series = ".".join(present.split(".")[:2]) != ".".join(expected.split(".")[:2])
        if present == ABSENT or series:
            found.append(f"{name}: recorded {expected}, present {present}")
    return found


def download(pointer: dict, destination: Path = MODEL_PATH) -> Path:
    url = pointer.get("url")
    if not url:
        raise ValueError("The pinned version carries no artifact URL.")

    response = requests.get(url, timeout=TIMEOUT)
    response.raise_for_status()

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)
    return destination


def main() -> int:
    pointer = read_pin()

    mismatched = blocking_differences(pointer)
    if mismatched:
        raise VersionMismatch(
            "The pinned artifact was written under different versions of "
            + "; ".join(mismatched)
            + ". Serving it here would predict from an estimator reconstructed by "
            "code that never fitted it."
        )

    path = download(pointer)
    print(
        f"{pointer['model']} version {pointer['version']} "
        f"at {pointer['revision'][:12]} -> {path}"
    )
    if "metric" in pointer:
        print(f"promoted on {pointer['metric']} {pointer['value']} "
              f"measured on held-out data {pointer['dataset']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
