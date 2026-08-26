import random

import pandas as pd
import pytest

from model.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET_COLUMN
from model.pipeline import build_pipeline

VALID_PAYLOAD = {
    "checkin_acc": "A11",
    "duration": 6,
    "credit_history": "A34",
    "purpose": "A43",
    "amount": 1169,
    "savings_acc": "A65",
    "present_emp_since": "A75",
    "installment_rate": 4,
    "personal_status": "A93",
    "other_debtors": "A101",
    "residing_since": 4,
    "property": "A121",
    "age": 67,
    "inst_plans": "A143",
    "housing": "A152",
    "num_credits": 2,
    "job": "A173",
    "dependents": 1,
    "telephone": "A192",
    "foreign_worker": "A201",
}

_CATEGORY_POOL = {
    "checkin_acc": ["A11", "A12", "A13", "A14"],
    "credit_history": ["A30", "A31", "A32", "A33", "A34"],
    "purpose": ["A40", "A41", "A43"],
    "savings_acc": ["A61", "A62", "A65"],
    "present_emp_since": ["A71", "A72", "A73", "A74", "A75"],
    "personal_status": ["A91", "A92", "A93"],
    "other_debtors": ["A101", "A102", "A103"],
    "property": ["A121", "A122", "A123"],
    "inst_plans": ["A141", "A142", "A143"],
    "housing": ["A151", "A152", "A153"],
    "job": ["A171", "A172", "A173"],
    "telephone": ["A191", "A192"],
    "foreign_worker": ["A201", "A202"],
}

_NUMERIC_RANGE = {
    "duration": (6, 48),
    "amount": (500, 10000),
    "installment_rate": (1, 4),
    "residing_since": (1, 4),
    "age": (19, 70),
    "num_credits": (1, 3),
    "dependents": (1, 2),
}


@pytest.fixture
def training_frame() -> pd.DataFrame:
    """
    A small synthetic dataset with the real schema and both classes present.

    Fitting on this keeps the suite offline: the real loader would hit UCI.
    """
    rng = random.Random(0)
    rows = []
    for i in range(60):
        row = {c: rng.choice(_CATEGORY_POOL[c]) for c in CATEGORICAL_FEATURES}
        row.update({c: rng.randint(*_NUMERIC_RANGE[c]) for c in NUMERICAL_FEATURES})
        row[TARGET_COLUMN] = 1 if i % 3 == 0 else 0
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def fitted_pipeline(training_frame):
    """The production pipeline, fitted on the synthetic frame."""
    pipeline = build_pipeline()
    pipeline.fit(
        training_frame.drop(TARGET_COLUMN, axis=1), training_frame[TARGET_COLUMN]
    )
    return pipeline


@pytest.fixture
def valid_payload() -> dict:
    return dict(VALID_PAYLOAD)
