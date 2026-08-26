from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class CreditApplication(BaseModel):
    """
    Schema representing the input data required for a credit risk prediction.
    Field names must match the columns expected by the trained model pipeline.
    """
    checkin_acc: str = Field(..., title="Checking Account", description="Status of existing checking account (e.g., A11, A12)")
    duration: int = Field(..., gt=0, title="Duration", description="Duration of credit in months")
    credit_history: str = Field(..., title="Credit History", description="Credit history (e.g., A30-A34)")
    purpose: str = Field(..., title="Purpose", description="Purpose of loan (e.g., A40-A410)")
    amount: int = Field(..., gt=0, title="Credit Amount", description="Credit amount in DM")
    savings_acc: str = Field(..., title="Savings Account", description="Savings account/bonds (e.g., A61-A65)")
    present_emp_since: str = Field(..., title="Employment Since", description="Present employment since (e.g., A71-A75)")
    installment_rate: int = Field(..., ge=1, le=4, title="Installment Rate", description="Installment rate in percentage of disposable income")
    personal_status: str = Field(..., title="Personal Status", description="Personal status and sex (e.g., A91-A95)")
    other_debtors: str = Field(..., title="Other Debtors", description="Other debtors / guarantors (e.g., A101-A103)")
    residing_since: int = Field(..., gt=0, title="Residing Since", description="Present residence since (years)")
    property: str = Field(..., title="Property", description="Property magnitude (e.g., A121-A124)")
    age: int = Field(..., ge=18, title="Age", description="Age in years")
    inst_plans: str = Field(..., title="Installment Plans", description="Other installment plans (e.g., A141-A143)")
    housing: str = Field(..., title="Housing", description="Housing status (e.g., A151-A153)")
    num_credits: int = Field(..., gt=0, title="Number of Credits", description="Number of existing credits at this bank")
    job: str = Field(..., title="Job", description="Job qualification (e.g., A171-A174)")
    dependents: int = Field(..., gt=0, title="Dependents", description="Number of people being liable to provide maintenance for")
    telephone: str = Field(..., title="Telephone", description="Telephone status (e.g., A191, A192)")
    foreign_worker: str = Field(..., title="Foreign Worker", description="Foreign worker status (e.g., A201, A202)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )

class PredictionResponse(BaseModel):
    """
    The scoring result, including why it came out that way.

    model_config disables Pydantic's protected "model_" namespace: the field is
    called model_version because that is what it is, and the warning it would
    otherwise raise is about a name collision that does not exist here.
    """

    model_config = ConfigDict(protected_namespaces=())

    decision_id: int = Field(..., description="Identifier of the stored decision")
    risk_class: int = Field(..., description="0 = Good Credit (No Risk), 1 = Bad Credit (Risk)")
    risk_label: str = Field(..., description="Human readable label: 'Low Risk' or 'High Risk'")
    probability: float = Field(..., description="Probability of the applicant defaulting")
    threshold: float = Field(..., description="Probability above which the applicant is flagged")
    model_version: str = Field(..., description="Content hash of the artifact that decided")
    contributions: dict[str, float] = Field(
        ..., description="Per-field contribution to the log-odds of default"
    )


class DecisionRecord(BaseModel):
    """One entry of the audit log."""

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: int
    created_at: datetime
    model_version: str
    threshold: float
    risk_class: int
    probability: float
    application: dict
    contributions: dict[str, float]
    defaulted: int | None = Field(
        None, description="Ground truth once known: 1 if the applicant defaulted"
    )
    outcome_recorded_at: datetime | None = None


class DecisionPage(BaseModel):
    """A page of the audit log, seeked by key rather than by offset."""

    items: list[DecisionRecord]
    next_cursor: int | None = Field(
        None, description="Pass as ?cursor= to fetch the following page; null at the end"
    )


class OutcomeRequest(BaseModel):
    """Ground truth reported back once the loan resolves."""

    defaulted: bool = Field(..., description="True if the applicant ended up defaulting")


class PortfolioSummary(BaseModel):
    """
    Aggregate figures over the decision log.

    Realised cost covers only the decisions with a recorded outcome: the cost
    matrix needs to know what actually happened, so a service with no feedback
    loop cannot report what it is costing.
    """

    total: int
    approved: int
    rejected: int
    approval_rate: float
    mean_probability: float
    outcomes_recorded: int
    false_negatives: int
    false_positives: int
    realised_cost: int
