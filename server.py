"""CreditWise Loan System - web server.

Serves the designed front end (templates/index.html) and a JSON prediction API.

Run:   python server.py      then open http://127.0.0.1:5000
Needs: loan_model.pkl  (create it with `python train_model.py`)
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE = Path(__file__).parent
bundle = joblib.load(BASE / "loan_model.pkl")
model, metrics, options = bundle["model"], bundle["metrics"], bundle["options"]

NUM_COLS = [
    "Applicant_Income", "Coapplicant_Income", "Age", "Dependents", "Credit_Score",
    "Existing_Loans", "DTI_Ratio", "Savings", "Collateral_Value", "Loan_Amount", "Loan_Term",
]
CAT_COLS = ["Employment_Status", "Marital_Status", "Loan_Purpose",
            "Property_Area", "Gender", "Employer_Category", "Education_Level"]
ALL_COLS = NUM_COLS + CAT_COLS

LABELS = {
    "Applicant_Income": "Applicant income", "Coapplicant_Income": "Co-applicant income",
    "Age": "Age", "Dependents": "Dependents", "Credit_Score": "Credit score",
    "Existing_Loans": "Existing loans", "DTI_Ratio": "Debt-to-income ratio",
    "Savings": "Savings", "Collateral_Value": "Collateral value", "Loan_Amount": "Loan amount",
    "Loan_Term": "Loan term", "Employment_Status": "Employment status",
    "Marital_Status": "Marital status", "Loan_Purpose": "Loan purpose",
    "Property_Area": "Property area", "Gender": "Gender",
    "Employer_Category": "Employer category", "Education_Level": "Education level",
}

# --- Explanation setup -------------------------------------------------------
# For logistic regression, each feature's push on the decision is
# coef * (applicant_value - average_applicant_value) in scaled space.
_pre, _scale, _clf = (model.named_steps[k] for k in ("preprocess", "scale", "clf"))
_feat_names = _pre.get_feature_names_out()
_coef = _clf.coef_[0]

_train = pd.read_csv(BASE / "2 loan_approval_data.csv")
_train = _train.dropna(subset=["Loan_Approved"])[ALL_COLS]
_mean_scaled = np.asarray(_scale.transform(_pre.transform(_train))).mean(axis=0)


def _source_column(feat: str) -> str:
    name = feat.split("__", 1)[1]
    for col in ALL_COLS:
        if name == col or name.startswith(col + "_"):
            return col
    return name


_feat_source = [_source_column(f) for f in _feat_names]

app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html", options=options, metrics=metrics)


@app.post("/api/predict")
def predict():
    data = request.get_json(force=True)
    try:
        row = {c: float(data[c]) for c in NUM_COLS}
        row.update({c: str(data[c]) for c in CAT_COLS})
    except (KeyError, TypeError, ValueError) as e:
        return jsonify(error=f"Invalid or missing field: {e}"), 400

    df = pd.DataFrame([row])
    prob = float(model.predict_proba(df)[0, 1])

    x = np.asarray(_scale.transform(_pre.transform(df)))[0]
    push = _coef * (x - _mean_scaled)
    by_col: dict[str, float] = {}
    for col, p in zip(_feat_source, push):
        by_col[col] = by_col.get(col, 0.0) + float(p)

    factors = sorted(
        ({"feature": LABELS.get(c, c), "impact": round(v, 3)} for c, v in by_col.items()),
        key=lambda f: abs(f["impact"]), reverse=True,
    )[:6]

    return jsonify(probability=prob, approved=prob >= 0.5, factors=factors)


if __name__ == "__main__":
    app.run(debug=False, port=5000)
