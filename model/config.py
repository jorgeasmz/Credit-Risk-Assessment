"""
Dataset layout, feature groups and artifact location.
"""

from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
MODEL_PATH = MODEL_DIR / "credit_risk_model.joblib"

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "statlog/german/german.data"
)

# The raw file has no header row.
COLUMNS = [
    "checkin_acc",        # Status of existing checking account
    "duration",           # Duration in months
    "credit_history",     # Credit history
    "purpose",            # Purpose of the loan
    "amount",             # Credit amount
    "savings_acc",        # Savings account / bonds
    "present_emp_since",  # Present employment since
    "installment_rate",   # Installment rate as % of disposable income
    "personal_status",    # Personal status and sex
    "other_debtors",      # Other debtors / guarantors
    "residing_since",     # Present residence since
    "property",           # Property
    "age",                # Age in years
    "inst_plans",         # Other installment plans
    "housing",            # Housing
    "num_credits",        # Existing credits at this bank
    "job",                # Job
    "dependents",         # People liable to provide maintenance for
    "telephone",          # Telephone
    "foreign_worker",     # Foreign worker
    "status",             # Target
]

TARGET_COLUMN = "status"

# Upstream encodes 1 = good, 2 = bad; we model risk, so bad is the positive class.
TARGET_MAPPING = {1: 0, 2: 1}

CATEGORICAL_FEATURES = [
    "checkin_acc", "credit_history", "purpose", "savings_acc",
    "present_emp_since", "personal_status", "other_debtors",
    "property", "inst_plans", "housing", "job", "telephone", "foreign_worker",
]

NUMERICAL_FEATURES = [
    "duration", "amount", "installment_rate", "residing_since",
    "age", "num_credits", "dependents",
]

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100

# UCI cost matrix: a missed default costs five times a rejected good applicant.
COST_FALSE_NEGATIVE = 5
COST_FALSE_POSITIVE = 1

# Chosen by the cost sweep in evaluate.py, not left at the 0.5 default.
DECISION_THRESHOLD = 0.45
