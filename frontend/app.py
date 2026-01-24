import streamlit as st
import requests
import os
import json

# --- Configuration ---
API_URL = os.environ.get("API_URL", "http://localhost:8000")

# Ensure the URL points to the /predict endpoint
if not API_URL.endswith("/predict"):
    API_URL = API_URL.rstrip("/") + "/predict"

st.set_page_config(
    page_title="Credit Risk Dashboard",
    page_icon="🏦",
    layout="wide"
)

# --- Header ---
st.title("🏦 AI Credit Risk Assessment")
st.markdown(f"**Backend Status:** Connecting to `{API_URL}`")
st.markdown("---")

# --- Input Form ---
with st.form("prediction_form"):
    st.header("Applicant Details")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Financial Status")
        checkin_acc = st.selectbox(
            "Checking Account Balance",
            options=["A11", "A12", "A13", "A14"],
            help="A11: < 0 DM | A12: 0-200 DM | A13: >= 200 DM | A14: No Account",
            index=0
        )
        savings_acc = st.selectbox(
            "Savings Account",
            options=["A61", "A62", "A63", "A64", "A65"],
            help="A61: < 100 DM | A65: Unknown/No Savings"
        )
        credit_history = st.selectbox(
            "Credit History",
            options=["A30", "A31", "A32", "A33", "A34"],
            help="A34: Critical account/Other credits existing"
        )
        property_val = st.selectbox(
            "Property Ownership",
            options=["A121", "A122", "A123", "A124"],
            help="A121: Real Estate | A124: Unknown/No Property"
        )
        other_debtors = st.selectbox("Guarantors / Debtors", ["A101", "A102", "A103"])

    with col2:
        st.subheader("Loan Details")
        amount = st.number_input("Credit Amount (DM)", min_value=100, max_value=20000, value=2000, step=100)
        duration = st.number_input("Duration (Months)", min_value=4, max_value=72, value=24, step=1)
        purpose = st.selectbox(
            "Purpose",
            options=["A40", "A41", "A42", "A43", "A44", "A45", "A46", "A48", "A49", "A410"],
            format_func=lambda x: f"{x} (Car/Furniture/Radio/etc)"  # Simple formatting helper
        )
        installment_rate = st.slider("Installment Rate (% of Income)", 1, 4, 3)
        inst_plans = st.selectbox("Other Installment Plans", ["A141", "A142", "A143"])

    with col3:
        st.subheader("Personal Info")
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        job = st.selectbox("Job Qualification", ["A171", "A172", "A173", "A174"])
        housing = st.selectbox("Housing", ["A151", "A152", "A153"], help="Rent | Own | Free")
        personal_status = st.selectbox("Personal Status", ["A91", "A92", "A93", "A94", "A95"])
        foreign_worker = st.radio("Foreign Worker?", ["A201", "A202"], index=0, help="A201: Yes, A202: No")
        
        with st.expander("Additional Details"):
            residing_since = st.slider("Residing Since (Years)", 1, 4, 2)
            num_credits = st.number_input("Existing Credits", 1, 10, 1)
            dependents = st.number_input("dependents", 1, 5, 1)
            present_emp_since = st.selectbox("Employed Since", ["A71", "A72", "A73", "A74", "A75"])
            telephone = st.selectbox("Telephone Registered?", ["A191", "A192"])

    # Submit Button
    submitted = st.form_submit_button("Analyze Risk", type="primary")

# --- Submission Logic ---
if submitted:
    # 1. Construct Payload

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
        "foreign_worker": foreign_worker
    }

    # 2. Visual Feedback
    with st.spinner("Consulting AI Model..."):
        try:
            # 3. Send Request to API
            response = requests.post(f"{API_URL}/predict", json=payload)
            
            # 4. Handle Response
            if response.status_code == 200:
                result = response.json()
                risk_class = result["risk_class"]
                probability = result["probability"]

                st.success("Analysis Complete")
                
                # Display Metrics
                col_res1, col_res2 = st.columns(2)
                
                with col_res1:
                    if risk_class == 1:
                        st.error(f"⚠️ Recommendation: REJECT")
                        st.metric("Risk Label", "High Risk")
                    else:
                        st.success(f"✅ Recommendation: APPROVE")
                        st.metric("Risk Label", "Low Risk")
                
                with col_res2:
                    st.metric("Probability of Default", f"{probability:.2%}")
                    st.progress(probability)
                
                # Debug info
                with st.expander("See Raw API Response"):
                    st.json(result)

            else:
                st.error(f"API Error: {response.status_code}")
                st.text(response.text)
                
        except requests.exceptions.ConnectionError:
            st.error("🚨 Connection Error: Could not connect to backend.")
            st.info(f"Ensure the API is running at: {API_URL}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")