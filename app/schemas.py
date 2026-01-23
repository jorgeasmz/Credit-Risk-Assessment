from pydantic import BaseModel, Field

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
    age: int = Field(..., gt=18, title="Age", description="Age in years")
    inst_plans: str = Field(..., title="Installment Plans", description="Other installment plans (e.g., A141-A143)")
    housing: str = Field(..., title="Housing", description="Housing status (e.g., A151-A153)")
    num_credits: int = Field(..., gt=0, title="Number of Credits", description="Number of existing credits at this bank")
    job: str = Field(..., title="Job", description="Job qualification (e.g., A171-A174)")
    dependents: int = Field(..., gt=0, title="Dependents", description="Number of people being liable to provide maintenance for")
    telephone: str = Field(..., title="Telephone", description="relephone status (e.g., A191, A192)")
    foreign_worker: str = Field(..., title="Foreign Worker", description="Foreign worker status (e.g., A201, A202)")

    class Config:
        jason_schema_extra = {
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
                "foreign_worker": "A201"
            }
        }

class PredictionResponse(BaseModel):
    """
    Schema representing the output of the prediction endpoint.
    """
    risk_class: int = Field(..., description="0 = Good Credit (No Risk), 1 = Bad Credit (Risk)")
    risk_label: str = Field(..., description="Human readable label: 'Low Risk' or 'High Risk'")
    probability: float = Field(..., description="Probability of the applicant defaulting (0.0 to 1.0)")