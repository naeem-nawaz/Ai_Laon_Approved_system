"""CreditWise Loan System - web front end.

Run:  streamlit run app.py
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_FILE = Path(__file__).parent / "loan_model.pkl"

st.set_page_config(page_title="CreditWise Loan System", page_icon="🏦", layout="wide")


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_FILE)


if not MODEL_FILE.exists():
    st.error("Model file not found. Run `python train_model.py` first, then refresh this page.")
    st.stop()

bundle = load_bundle()
model, metrics, opts = bundle["model"], bundle["metrics"], bundle["options"]

# ---------- Header ----------
st.title("🏦 CreditWise Loan System")
st.caption("Fill in the applicant's details to get an instant loan decision. "
           "Final approval still requires human verification.")

with st.sidebar:
    st.header("About the model")
    st.write("Logistic Regression trained on historical loan applications.")
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{metrics['accuracy']:.0%}")
    c2.metric("F1 score", f"{metrics['f1']:.0%}")
    c1.metric("Precision", f"{metrics['precision']:.0%}")
    c2.metric("Recall", f"{metrics['recall']:.0%}")
    st.caption("Scores measured on a held-out 20% test set.")

# ---------- Form ----------
with st.form("applicant"):
    st.subheader("👤 Personal details")
    a, b, c, d = st.columns(4)
    age = a.number_input("Age", 18, 75, 35)
    gender = b.selectbox("Gender", opts["Gender"])
    marital = c.selectbox("Marital status", opts["Marital_Status"])
    dependents = d.number_input("Dependents", 0, 10, 1)

    a, b, c, d = st.columns(4)
    education = a.selectbox("Education level", opts["Education_Level"])
    employment = b.selectbox("Employment status", opts["Employment_Status"])
    employer = c.selectbox("Employer category", opts["Employer_Category"])
    area = d.selectbox("Property area", opts["Property_Area"])

    st.subheader("💰 Financial details")
    a, b, c, d = st.columns(4)
    income = a.number_input("Applicant income", 0, 1_000_000, 10_000, step=500)
    co_income = b.number_input("Co-applicant income", 0, 1_000_000, 5_000, step=500)
    savings = c.number_input("Savings", 0, 10_000_000, 10_000, step=500)
    collateral = d.number_input("Collateral value", 0, 10_000_000, 25_000, step=500)

    st.subheader("📊 Credit & loan details")
    a, b, c = st.columns(3)
    credit = a.slider("Credit score", 300, 900, 680)
    dti = b.slider("Debt-to-income ratio", 0.0, 1.0, 0.35, 0.01,
                   help="Monthly debt payments ÷ monthly income")
    existing = c.number_input("Existing loans", 0, 10, 1)

    a, b, c = st.columns(3)
    amount = a.number_input("Loan amount", 500, 10_000_000, 20_000, step=500)
    term = b.selectbox("Loan term (months)", [12, 24, 36, 48, 60, 72, 84], index=3)
    purpose = c.selectbox("Loan purpose", opts["Loan_Purpose"])

    submitted = st.form_submit_button("Check eligibility", type="primary", use_container_width=True)

# ---------- Prediction ----------
if submitted:
    row = pd.DataFrame([{
        "Applicant_Income": income, "Coapplicant_Income": co_income, "Age": age,
        "Dependents": dependents, "Credit_Score": credit, "Existing_Loans": existing,
        "DTI_Ratio": dti, "Savings": savings, "Collateral_Value": collateral,
        "Loan_Amount": amount, "Loan_Term": term,
        "Employment_Status": employment, "Marital_Status": marital, "Loan_Purpose": purpose,
        "Property_Area": area, "Gender": gender, "Employer_Category": employer,
        "Education_Level": education,
    }])

    prob = float(model.predict_proba(row)[0, 1])
    approved = prob >= 0.5

    st.divider()
    left, right = st.columns([2, 3])
    with left:
        if approved:
            st.success("### ✅ Loan likely APPROVED")
        else:
            st.error("### ❌ Loan likely REJECTED")
        st.metric("Approval probability", f"{prob:.1%}")
        st.progress(prob)

    with right:
        st.markdown("**Key factors**")
        notes = []
        notes.append(("🟢" if credit >= 700 else "🟡" if credit >= 650 else "🔴",
                      f"Credit score {credit} — strongest positive factor"))
        notes.append(("🟢" if dti <= 0.3 else "🟡" if dti <= 0.45 else "🔴",
                      f"Debt-to-income {dti:.2f} — lower is better"))
        notes.append(("🟢" if amount <= income * 2 else "🟡" if amount <= income * 3 else "🔴",
                      f"Loan amount {amount:,} vs income {income:,}"))
        for icon, text in notes:
            st.write(f"{icon} {text}")
        st.caption("This is a decision-support tool. A loan officer should review before final approval.")
