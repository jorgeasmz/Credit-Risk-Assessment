import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")

# Free Render instances spin down when idle and take up to a minute to wake,
# so the first request of the day needs far more than a default timeout.
REQUEST_TIMEOUT = int(os.environ.get("API_TIMEOUT", "90"))

# Codes come straight from the UCI documentation; the raw values mean nothing
# to anyone filling in the form.
PURPOSE_LABELS = {
    "A40": "A40 - New car",
    "A41": "A41 - Used car",
    "A42": "A42 - Furniture / equipment",
    "A43": "A43 - Radio / television",
    "A44": "A44 - Domestic appliances",
    "A45": "A45 - Repairs",
    "A46": "A46 - Education",
    "A48": "A48 - Retraining",
    "A49": "A49 - Business",
    "A410": "A410 - Other",
}

# How many fields the waterfall shows before collapsing the rest.
TOP_CONTRIBUTIONS = 8

st.set_page_config(page_title="Credit Risk Dashboard", page_icon="🏦", layout="wide")


def api_key() -> str:
    """The service key, from Streamlit secrets or the environment."""
    try:
        return st.secrets["API_KEY"]
    except (FileNotFoundError, KeyError):
        return os.environ.get("API_KEY", "")


def call_api(method: str, path: str, **kwargs):
    """
    One place for the header, the timeout and the failure modes.

    Returns (payload, error). Callers render one or the other; nothing raises
    into the Streamlit script.
    """
    key = api_key()
    if not key:
        return None, "No API key configured. Set API_KEY in the app secrets."

    try:
        response = requests.request(
            method,
            f"{API_URL}{path}",
            headers={"X-API-Key": key},
            timeout=REQUEST_TIMEOUT,
            **kwargs,
        )
    except requests.exceptions.Timeout:
        return None, (
            f"The API did not answer within {REQUEST_TIMEOUT}s. A free-tier "
            "backend may be waking up from idle. Try again in a moment."
        )
    except requests.exceptions.ConnectionError:
        return None, f"Could not reach the API at {API_URL}."

    if response.status_code == 401:
        return None, "The API rejected the key."
    if response.status_code >= 400:
        return None, f"API error {response.status_code}: {response.text[:200]}"

    return response.json(), None


def contribution_waterfall(contributions: dict) -> go.Figure:
    """
    Shows how each field moved the decision.

    Values are in log-odds, which is the scale the model actually adds on. They
    are not percentages and do not sum to the probability, so the axis says so
    rather than inviting the wrong reading.
    """
    items = list(contributions.items())
    head, tail = items[:TOP_CONTRIBUTIONS], items[TOP_CONTRIBUTIONS:]
    if tail:
        head.append((f"other ({len(tail)} fields)", sum(v for _, v in tail)))

    labels = [name for name, _ in head]
    values = [value for _, value in head]

    figure = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["relative"] * len(head),
            x=labels,
            y=values,
            connector={"line": {"color": "rgba(120,120,120,0.4)"}},
            increasing={"marker": {"color": "#c0392b"}},
            decreasing={"marker": {"color": "#1e8449"}},
        )
    )
    figure.update_layout(
        title="Why this decision (log-odds of default)",
        yaxis_title="Contribution",
        showlegend=False,
        height=420,
        margin={"t": 60, "b": 40},
    )
    return figure


def render_assessment() -> None:
    with st.form("prediction_form"):
        st.subheader("Applicant Details")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Financial Status**")
            checkin_acc = st.selectbox(
                "Checking Account Balance",
                ["A11", "A12", "A13", "A14"],
                help="A11: < 0 DM | A12: 0-200 DM | A13: >= 200 DM | A14: No Account",
            )
            savings_acc = st.selectbox(
                "Savings Account",
                ["A61", "A62", "A63", "A64", "A65"],
                help="A61: < 100 DM | A65: Unknown/No Savings",
            )
            credit_history = st.selectbox(
                "Credit History",
                ["A30", "A31", "A32", "A33", "A34"],
                help="A34: Critical account/Other credits existing",
            )
            property_val = st.selectbox(
                "Property Ownership", ["A121", "A122", "A123", "A124"]
            )
            other_debtors = st.selectbox("Guarantors / Debtors", ["A101", "A102", "A103"])

        with col2:
            st.markdown("**Loan Details**")
            amount = st.number_input("Credit Amount (DM)", 100, 20000, 2000, step=100)
            duration = st.number_input("Duration (Months)", 4, 72, 24)
            purpose = st.selectbox(
                "Purpose",
                list(PURPOSE_LABELS),
                format_func=lambda code: PURPOSE_LABELS.get(code, code),
            )
            installment_rate = st.slider("Installment Rate (% of Income)", 1, 4, 3)
            inst_plans = st.selectbox("Other Installment Plans", ["A141", "A142", "A143"])

        with col3:
            st.markdown("**Personal Info**")
            age = st.number_input("Age", 18, 100, 30)
            job = st.selectbox("Job Qualification", ["A171", "A172", "A173", "A174"])
            housing = st.selectbox("Housing", ["A151", "A152", "A153"], help="Rent | Own | Free")
            personal_status = st.selectbox(
                "Personal Status", ["A91", "A92", "A93", "A94", "A95"]
            )
            foreign_worker = st.radio("Foreign Worker?", ["A201", "A202"], horizontal=True)
            with st.expander("Additional Details"):
                residing_since = st.slider("Residing Since (Years)", 1, 4, 2)
                num_credits = st.number_input("Existing Credits", 1, 10, 1)
                dependents = st.number_input("Dependents", 1, 5, 1)
                present_emp_since = st.selectbox(
                    "Employed Since", ["A71", "A72", "A73", "A74", "A75"]
                )
                telephone = st.selectbox("Telephone Registered?", ["A191", "A192"])

        submitted = st.form_submit_button("Analyze Risk", type="primary")

    if not submitted:
        return

    payload = {
        "checkin_acc": checkin_acc,
        "duration": duration,
        "credit_history": credit_history,
        "purpose": purpose,
        "amount": amount,
        "savings_acc": savings_acc,
        "present_emp_since": present_emp_since,
        "installment_rate": installment_rate,
        "personal_status": personal_status,
        "other_debtors": other_debtors,
        "residing_since": residing_since,
        "property": property_val,
        "age": age,
        "inst_plans": inst_plans,
        "housing": housing,
        "num_credits": num_credits,
        "job": job,
        "dependents": dependents,
        "telephone": telephone,
        "foreign_worker": foreign_worker,
    }

    with st.spinner("Scoring the application..."):
        result, error = call_api("post", "/predict", json=payload)

    if error:
        st.error(error)
        return

    st.divider()
    left, right = st.columns([1, 2])

    with left:
        if result["risk_class"] == 1:
            st.error("Recommendation: REJECT")
        else:
            st.success("Recommendation: APPROVE")
        st.metric("Probability of Default", f"{result['probability']:.2%}")
        st.caption(
            f"Threshold {result['threshold']:.2f} · model `{result['model_version']}` · "
            f"decision #{result['decision_id']}"
        )

    with right:
        st.plotly_chart(contribution_waterfall(result["contributions"]), width="stretch")

    with st.expander("Raw API response"):
        st.json(result)


def render_decision_log() -> None:
    st.subheader("Decision log")
    st.caption(
        "Every scoring is stored with its inputs, its explanation and the artifact "
        "that produced it, so any decision can be reproduced after the fact."
    )

    page, error = call_api("get", "/decisions?limit=25")
    if error:
        st.error(error)
        return

    items = page["items"]
    if not items:
        st.info("No decisions recorded yet.")
        return

    table = pd.DataFrame(
        [
            {
                "id": item["id"],
                "when": item["created_at"],
                "decision": "REJECT" if item["risk_class"] else "APPROVE",
                "probability": item["probability"],
                "model": item["model_version"],
                "outcome": {None: "unknown", 0: "repaid", 1: "defaulted"}[item["defaulted"]],
            }
            for item in items
        ]
    )
    st.dataframe(table, width="stretch", hide_index=True)

    st.markdown("**Record an outcome**")
    st.caption(
        "Cost cannot be measured without knowing what actually happened. This is "
        "the feedback loop that turns the log into an evaluation set."
    )
    col1, col2, col3 = st.columns([1, 1, 2])
    decision_id = col1.number_input("Decision id", min_value=1, step=1, value=int(items[0]["id"]))
    defaulted = col2.selectbox("Outcome", ["repaid", "defaulted"])
    if col3.button("Save outcome"):
        _, outcome_error = call_api(
            "post",
            f"/decisions/{int(decision_id)}/outcome",
            json={"defaulted": defaulted == "defaulted"},
        )
        if outcome_error:
            st.error(outcome_error)
        else:
            st.success(f"Outcome recorded for decision #{int(decision_id)}.")
            st.rerun()


def render_portfolio() -> None:
    st.subheader("Portfolio")

    summary, error = call_api("get", "/summary")
    if error:
        st.error(error)
        return

    row = st.columns(4)
    row[0].metric("Decisions", summary["total"])
    row[1].metric("Approval rate", f"{summary['approval_rate']:.1%}")
    row[2].metric("Mean default probability", f"{summary['mean_probability']:.2%}")
    row[3].metric("Outcomes known", summary["outcomes_recorded"])

    st.divider()

    if not summary["outcomes_recorded"]:
        st.info(
            "Realised cost needs outcomes. Record a few in the decision log and this "
            "section starts reporting what the model's mistakes actually cost."
        )
        return

    cost = st.columns(3)
    cost[0].metric("Missed defaults", summary["false_negatives"], help="Approved, then defaulted")
    cost[1].metric("Rejected good applicants", summary["false_positives"])
    cost[2].metric("Realised cost", summary["realised_cost"])
    st.caption(
        "Priced with the UCI cost matrix: a missed default costs 5, a rejected "
        "good applicant costs 1."
    )


st.title("AI Credit Risk Assessment")
st.caption(f"Backend: `{API_URL}`")

assess_tab, log_tab, portfolio_tab = st.tabs(["Assess", "Decision log", "Portfolio"])
with assess_tab:
    render_assessment()
with log_tab:
    render_decision_log()
with portfolio_tab:
    render_portfolio()
