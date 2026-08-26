from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from model.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from model.pipeline import build_classifier, build_forest, build_pipeline


def test_production_classifier_is_the_interpretable_one():
    assert isinstance(build_classifier(), LogisticRegression)
    assert isinstance(build_forest(), RandomForestClassifier)


def test_both_classifiers_correct_for_the_class_imbalance():
    assert build_classifier().class_weight == "balanced"
    assert build_forest().class_weight == "balanced"


def test_preprocessing_lives_inside_the_pipeline(fitted_pipeline):
    """This is what makes train/serve skew impossible: raw fields go in."""
    steps = dict(fitted_pipeline.named_steps)

    assert "preprocessor" in steps
    transformers = dict(
        (name, columns) for name, _, columns in steps["preprocessor"].transformers
    )
    assert transformers["num"] == NUMERICAL_FEATURES
    assert transformers["cat"] == CATEGORICAL_FEATURES


def test_pipeline_accepts_an_injected_classifier(training_frame):
    pipeline = build_pipeline(build_forest())

    assert isinstance(pipeline.named_steps["classifier"], RandomForestClassifier)
