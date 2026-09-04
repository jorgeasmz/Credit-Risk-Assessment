import json

import pandas as pd
import pytest

from model.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET_COLUMN
from model.release import (
    build_card,
    holdout,
    holdout_digest,
    measure,
    total_cost,
)


@pytest.fixture
def raw_file(tmp_path, training_frame):
    """The synthetic frame in the layout the loader expects: no header, 1 and 2."""
    frame = training_frame.copy()
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].map({0: 1, 1: 2})
    columns = CATEGORICAL_FEATURES + NUMERICAL_FEATURES + [TARGET_COLUMN]
    path = tmp_path / "german.data"
    frame[columns].to_csv(path, sep=" ", header=False, index=False)
    return str(path)


def test_the_cost_matrix_weights_a_missed_default_five_times(training_frame):
    actual = [0, 0, 1, 1]

    # One false positive, no false negative.
    assert total_cost(actual, [0, 1, 1, 1]) == 1
    # One false negative, no false positive.
    assert total_cost(actual, [0, 0, 0, 1]) == 5
    assert total_cost(actual, [0, 0, 1, 1]) == 0


def test_the_measurement_reports_the_metric_the_gate_turns_on(
    fitted_pipeline, training_frame
):
    features = training_frame.drop(TARGET_COLUMN, axis=1)
    target = training_frame[TARGET_COLUMN]

    record = measure(fitted_pipeline, features, target)

    assert record["metric"] == "total_cost"
    assert isinstance(record["value"], int)
    assert record["rows"] == len(target)
    assert 0.0 <= record["roc_auc"] <= 1.0
    assert 0.0 <= record["accuracy"] <= 1.0


def test_the_digest_is_stable_across_calls(training_frame):
    features = training_frame.drop(TARGET_COLUMN, axis=1)
    target = training_frame[TARGET_COLUMN]

    assert holdout_digest(features, target) == holdout_digest(features, target)


def test_the_digest_does_not_depend_on_row_order(training_frame):
    """The split returns rows in one order; a reordering is the same held-out data."""
    features = training_frame.drop(TARGET_COLUMN, axis=1)
    target = training_frame[TARGET_COLUMN]
    shuffled = features.sample(frac=1, random_state=7)

    assert holdout_digest(shuffled, target[shuffled.index]) == holdout_digest(
        features, target
    )


def test_a_changed_value_changes_the_digest(training_frame):
    features = training_frame.drop(TARGET_COLUMN, axis=1)
    target = training_frame[TARGET_COLUMN]
    altered = features.copy()
    altered.loc[altered.index[0], "amount"] += 1

    assert holdout_digest(altered, target) != holdout_digest(features, target)


def test_a_changed_label_changes_the_digest(training_frame):
    features = training_frame.drop(TARGET_COLUMN, axis=1)
    target = training_frame[TARGET_COLUMN]
    flipped = target.copy()
    flipped.iloc[0] = 1 - flipped.iloc[0]

    assert holdout_digest(features, flipped) != holdout_digest(features, target)


def test_the_split_is_reproducible(raw_file):
    """What lets a digest identify the held-out rows rather than describe one run."""
    first_x, first_y = holdout(raw_file)
    second_x, second_y = holdout(raw_file)

    assert holdout_digest(first_x, first_y) == holdout_digest(second_x, second_y)
    assert list(first_x.index) == list(second_x.index)


def test_the_split_holds_out_the_configured_share(raw_file):
    features, target = holdout(raw_file)

    assert len(features) == len(target) == 12  # 20% of the 60 synthetic rows


def test_the_card_reports_the_figures_from_the_record():
    record = {
        "metric": "total_cost", "value": 96, "rows": 200, "threshold": 0.45,
        "roc_auc": 0.8058, "accuracy": 0.72, "dataset": "ba349129c0b0c828",
        "artifact_sha256": "4d17fdeb3b1d", "environment": {},
    }

    card = build_card(record)

    assert "| Expected loss under the cost matrix | 96 |" in card
    assert "200 held-out applicants" in card
    assert "ba349129c0b0c828" in card
    # The environment and the artifact hash belong in the record, not the prose.
    assert "4d17fdeb3b1d" not in card


def test_the_record_written_by_the_command_round_trips(tmp_path):
    record = {"metric": "total_cost", "value": 96, "dataset": "abc"}
    path = tmp_path / "release.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

    assert json.loads(path.read_text()) == record


def test_an_empty_frame_is_not_measurable(fitted_pipeline, training_frame):
    """A holdout that lost its rows would otherwise report a cost of zero."""
    empty = training_frame.drop(TARGET_COLUMN, axis=1).iloc[:0]

    with pytest.raises(ValueError):
        measure(fitted_pipeline, empty, pd.Series(dtype=int))


def test_the_published_commit_is_written_back_into_the_record(monkeypatch, tmp_path):
    """The gate reads the score and the commit from one file, not from two commands."""
    import json as json_module

    from model import release

    path = tmp_path / "release.json"
    monkeypatch.setattr(release, "RELEASE_PATH", path)
    monkeypatch.setattr(release, "build_record", lambda: {
        "metric": "total_cost", "value": 96, "rows": 200, "threshold": 0.45,
        "roc_auc": 0.8, "accuracy": 0.72, "dataset": "abc", "environment": {},
        "artifact_sha256": "deadbeef",
    })
    monkeypatch.setattr(release, "publish", lambda record, repo: "f" * 40)
    monkeypatch.setattr("sys.argv", ["release", "--publish"])

    release.main()

    assert json_module.loads(path.read_text())["revision"] == "f" * 40
