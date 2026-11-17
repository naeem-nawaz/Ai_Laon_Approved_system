# CreditWise Loan Approval System

A machine learning system that predicts whether a loan application should be **approved** or **rejected**, built for a mid-sized bank that wants to replace slow, inconsistent manual verification with a fast, data-driven recommendation before final human review.

## Features

- Logistic Regression model trained on historical loan applications (~89% accuracy on a held-out test set)
- Web front end: applicant form, approval probability gauge, and a per-factor explanation of what pushed the decision towards approval or rejection
- Sample applicants (strong / average / high risk) for quick demos
- Jupyter notebook with EDA, preprocessing, and model training

## Project structure

| File | Purpose |
|---|---|
| `credit_wise.ipynb` | EDA, missing-value handling, encoding, correlation analysis, model training |
| `train_model.py` | Trains the full preprocessing + model pipeline and saves `loan_model.pkl` |
| `server.py` | Flask server: serves the web page and the `/api/predict` endpoint |
| `templates/index.html` | Front end (HTML/CSS/JS) |
| `app.py` | Alternative quick UI built with Streamlit |
| `2 loan_approval_data.csv` | Dataset (1,000 applicants) |

## Getting started

```bash
pip install -r requirements.txt
python train_model.py      # creates loan_model.pkl (already included)
python server.py           # open http://127.0.0.1:5000
```

Streamlit version:

```bash
streamlit run app.py
```

## API

`POST /api/predict` with a JSON body containing the applicant fields:

```json
{
  "Age": 35, "Dependents": 1, "Gender": "Male", "Marital_Status": "Married",
  "Education_Level": "Graduate", "Employment_Status": "Salaried",
  "Employer_Category": "Private", "Property_Area": "Urban",
  "Applicant_Income": 10000, "Coapplicant_Income": 5000, "Savings": 10000,
  "Collateral_Value": 25000, "Credit_Score": 680, "DTI_Ratio": 0.35,
  "Existing_Loans": 1, "Loan_Amount": 20000, "Loan_Term": 48, "Loan_Purpose": "Personal"
}
```

Response:

```json
{ "probability": 0.19, "approved": false, "factors": [{ "feature": "Credit score", "impact": 0.15 }] }
```

## Model

- Preprocessing: mean imputation (numeric), most-frequent imputation + one-hot encoding (categorical), ordinal encoding (education), standard scaling
- Classifier: Logistic Regression
- Strongest signals: **credit score** (positive) and **debt-to-income ratio** (negative)

> This is a decision-support tool. A loan officer should verify documents before the final decision.
