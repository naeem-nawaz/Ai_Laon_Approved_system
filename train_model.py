"""Train the CreditWise loan approval model and save it to loan_model.pkl.

Same steps as credit_wise.ipynb (impute -> encode -> scale -> Logistic Regression),
packed into one sklearn Pipeline so the web app can feed raw form values straight in.

Run:  python train_model.py
"""
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

DATA_FILE = "2 loan_approval_data.csv"
MODEL_FILE = "loan_model.pkl"

NUM_COLS = [
    "Applicant_Income", "Coapplicant_Income", "Age", "Dependents", "Credit_Score",
    "Existing_Loans", "DTI_Ratio", "Savings", "Collateral_Value", "Loan_Amount", "Loan_Term",
]
ONEHOT_COLS = ["Employment_Status", "Marital_Status", "Loan_Purpose",
               "Property_Area", "Gender", "Employer_Category"]
ORDINAL_COLS = ["Education_Level"]

df = pd.read_csv(DATA_FILE)
df = df.drop(columns="Applicant_ID")
df = df.dropna(subset=["Loan_Approved"])           # target must be known to train on

X = df[NUM_COLS + ONEHOT_COLS + ORDINAL_COLS]
y = (df["Loan_Approved"] == "Yes").astype(int)     # 1 = Approved, 0 = Rejected

preprocess = ColumnTransformer([
    ("num", SimpleImputer(strategy="mean"), NUM_COLS),
    ("onehot", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(drop="first", handle_unknown="ignore")),
    ]), ONEHOT_COLS),
    ("ordinal", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder()),
    ]), ORDINAL_COLS),
])

model = Pipeline([
    ("preprocess", preprocess),
    ("scale", StandardScaler(with_mean=False)),
    ("clf", LogisticRegression(max_iter=1000)),
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
}
print("Logistic Regression (test set)")
for k, v in metrics.items():
    print(f"  {k:<9} {v:.3f}")
print("  confusion matrix:\n", confusion_matrix(y_test, y_pred))

# Retrain on all data for the final model, keep test metrics for display in the app
model.fit(X, y)

options = {c: sorted(df[c].dropna().unique().tolist()) for c in ONEHOT_COLS + ORDINAL_COLS}
joblib.dump({"model": model, "metrics": metrics, "options": options}, MODEL_FILE)
print(f"\nSaved -> {MODEL_FILE}")
