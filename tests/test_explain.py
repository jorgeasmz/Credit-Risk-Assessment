import numpy as np

from model.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET_COLUMN
from model.explain import (
    _dense,
    build_background,
    build_explainer,
    explain,
    transformed_feature_owners,
)


def test_every_transformed_column_has_an_owner(fitted_pipeline, training_frame):
    features = training_frame.drop(TARGET_COLUMN, axis=1)
    transformed = build_background(fitted_pipeline, features)

    owners = transformed_feature_owners(fitted_pipeline)

    assert len(owners) == transformed.shape[1]


def test_owners_cover_every_original_field(fitted_pipeline):
    owners = set(transformed_feature_owners(fitted_pipeline))

    assert owners == set(NUMERICAL_FEATURES) | set(CATEGORICAL_FEATURES)


def test_explanation_names_the_fields_the_applicant_filled_in(
    fitted_pipeline, training_frame, valid_payload
):
    features = training_frame.drop(TARGET_COLUMN, axis=1)
    explainer = build_explainer(fitted_pipeline, build_background(fitted_pipeline, features))

    contributions = explain(fitted_pipeline, explainer, features.head(1))

    assert set(contributions) == set(NUMERICAL_FEATURES) | set(CATEGORICAL_FEATURES)
    assert all(isinstance(v, float) for v in contributions.values())


def test_contributions_add_up_to_the_model_output(fitted_pipeline, training_frame):
    """
    SHAP's additivity property: base value plus contributions equals the
    log-odds the model actually produced. If the one-hot columns were being
    aggregated wrongly, this is where it would show.
    """
    features = training_frame.drop(TARGET_COLUMN, axis=1)
    background = build_background(fitted_pipeline, features)
    explainer = build_explainer(fitted_pipeline, background)
    row = features.head(1)

    contributions = explain(fitted_pipeline, explainer, row)

    transformed = _dense(fitted_pipeline.named_steps["preprocessor"].transform(row))
    log_odds = fitted_pipeline.named_steps["classifier"].decision_function(transformed)[0]

    assert np.isclose(
        float(explainer.expected_value) + sum(contributions.values()), log_odds, atol=1e-6
    )


def test_the_same_application_is_always_explained_the_same_way(
    fitted_pipeline, training_frame
):
    """A decision that has to be defensible cannot be explained two ways."""
    features = training_frame.drop(TARGET_COLUMN, axis=1)
    explainer = build_explainer(fitted_pipeline, build_background(fitted_pipeline, features))
    row = features.head(1)

    assert explain(fitted_pipeline, explainer, row) == explain(
        fitted_pipeline, explainer, row
    )


def test_contributions_are_ordered_by_influence(fitted_pipeline, training_frame):
    features = training_frame.drop(TARGET_COLUMN, axis=1)
    explainer = build_explainer(fitted_pipeline, build_background(fitted_pipeline, features))

    values = list(explain(fitted_pipeline, explainer, features.head(1)).values())

    magnitudes = [abs(v) for v in values]
    assert magnitudes == sorted(magnitudes, reverse=True)
