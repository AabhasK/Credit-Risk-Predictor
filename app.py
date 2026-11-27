"""Local Gradio demo for the loan default model.

Trains the same pipeline as MLproject.ipynb (same data, labels, split and
preprocessing) at start-up, then lets you score a new applicant in the browser.
The notebook is not changed or imported.

Run:  python app.py   then open http://127.0.0.1:7860
"""
import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")  # silences a harmless joblib core-count warning on Windows

import pandas as pd
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ---- data: same columns, labels and split as the notebook
cols = ["status_checking", "duration_months", "credit_history", "purpose", "credit_amount",
        "savings", "employment_since", "installment_rate", "personal_status_sex",
        "other_debtors", "residence_since", "property", "age", "other_installment_plans",
        "housing", "existing_credits", "job", "num_dependents", "telephone", "foreign_worker"]

labels = {
    "status_checking": {"A11": "< 0 DM", "A12": "0 to 200 DM", "A13": ">= 200 DM", "A14": "no checking account"},
    "credit_history": {"A30": "no credits / all paid", "A31": "all paid at this bank",
                       "A32": "existing paid till now", "A33": "delays in the past",
                       "A34": "critical account"},
    "purpose": {"A40": "new car", "A41": "used car", "A42": "furniture", "A43": "radio/TV",
                "A44": "appliances", "A45": "repairs", "A46": "education", "A48": "retraining",
                "A49": "business", "A410": "other"},
    "savings": {"A61": "< 100 DM", "A62": "100 to 500 DM", "A63": "500 to 1000 DM",
                "A64": ">= 1000 DM", "A65": "unknown / none"},
    "employment_since": {"A71": "unemployed", "A72": "< 1 year", "A73": "1 to 4 years",
                         "A74": "4 to 7 years", "A75": ">= 7 years"},
    "personal_status_sex": {"A91": "male divorced", "A92": "female div/married", "A93": "male single",
                            "A94": "male married", "A95": "female single"},
    "other_debtors": {"A101": "none", "A102": "co-applicant", "A103": "guarantor"},
    "property": {"A121": "real estate", "A122": "savings/insurance", "A123": "car or other",
                 "A124": "unknown / none"},
    "other_installment_plans": {"A141": "bank", "A142": "stores", "A143": "none"},
    "housing": {"A151": "rent", "A152": "own", "A153": "for free"},
    "job": {"A171": "unskilled non-resident", "A172": "unskilled resident",
            "A173": "skilled employee", "A174": "management / self-employed"},
    "telephone": {"A191": "none", "A192": "yes"},
    "foreign_worker": {"A201": "yes", "A202": "no"},
}

raw = pd.read_csv("data/german.data", sep=" ", header=None)
X, y = raw.iloc[:, :20].copy(), (raw.iloc[:, 20] == 2).astype(int)
X.columns = cols
for col, mapping in labels.items():
    X[col] = X[col].map(mapping)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()


# ---- models: the five balanced pipelines from the notebook
def make_prep():
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])


spw = (y_train == 0).sum() / (y_train == 1).sum()
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=42),
    "XGBoost": XGBClassifier(scale_pos_weight=spw, eval_metric="logloss", random_state=42),
    "LightGBM": LGBMClassifier(class_weight="balanced", random_state=42, verbose=-1),
}
pipes = {name: Pipeline([("prep", make_prep()), ("clf", clf)]).fit(X_train, y_train)
         for name, clf in models.items()}
FINAL = "Logistic Regression"

# defaults = typical training applicant (median for numbers, most common for categories)
defaults = {c: (X_train[c].median() if c in num_cols else X_train[c].mode()[0]) for c in cols}

pretty = {
    "status_checking": "Checking account status", "duration_months": "Loan duration (months)",
    "credit_history": "Credit history", "purpose": "Purpose", "credit_amount": "Credit amount (DM)",
    "savings": "Savings", "employment_since": "Employed since", "installment_rate": "Installment rate (% of income, 1-4)",
    "personal_status_sex": "Personal status / sex", "other_debtors": "Other debtors / guarantors",
    "residence_since": "Years at residence (1-4)", "property": "Property", "age": "Age (years)",
    "other_installment_plans": "Other installment plans", "housing": "Housing",
    "existing_credits": "Existing credits at this bank", "job": "Job",
    "num_dependents": "Number of dependents", "telephone": "Telephone", "foreign_worker": "Foreign worker",
}


def predict(*values):
    row = pd.DataFrame([dict(zip(cols, values))])
    for c in num_cols:
        row[c] = pd.to_numeric(row[c])
    probs = {name: float(p.predict_proba(row)[0, 1]) for name, p in pipes.items()}
    p = probs[FINAL]
    verdict = "REJECT: likely to default" if p >= 0.5 else "APPROVE: unlikely to default"
    table = pd.DataFrame({
        "Model": list(probs),
        "P(default)": [f"{v:.0%}" for v in probs.values()],
        "Decision (>= 50% rejects)": ["reject" if v >= 0.5 else "approve" for v in probs.values()],
    })
    return f"## {verdict}\nFinal model ({FINAL}, balanced): P(default) = **{p:.0%}**", table


with gr.Blocks(title="Loan Default Prediction") as demo:
    gr.Markdown("# Loan Default Prediction\nUCI German Credit, balanced models trained on the same "
                "800-row training split as the notebook. Fields start at a typical applicant; change any and press Predict.")
    inputs = []
    with gr.Row():
        groups = [cols[:7], cols[7:14], cols[14:]]
        for group in groups:
            with gr.Column():
                for c in group:
                    if c in num_cols:
                        lo, hi = int(X[c].min()), int(X[c].max())
                        inputs.append(gr.Number(value=defaults[c], label=f"{pretty[c]}  [{lo}-{hi}]",
                                                minimum=lo, maximum=hi, precision=0))
                    else:
                        inputs.append(gr.Dropdown(list(labels[c].values()), value=defaults[c], label=pretty[c]))
    # keep inputs in the original column order for predict()
    order = [c for g in groups for c in g]
    assert order == cols
    btn = gr.Button("Predict", variant="primary")
    verdict = gr.Markdown()
    table = gr.Dataframe(label="All five balanced models", interactive=False)
    gr.Examples(
        examples=[[*[{"credit_amount": 9000, "duration_months": 36, "status_checking": "< 0 DM"}.get(c, defaults[c])
                     for c in cols]]],
        inputs=inputs, label="Risky example from the notebook (9000 DM, 36 months, checking < 0 DM)")
    btn.click(predict, inputs, [verdict, table])

if __name__ == "__main__":
    demo.launch()
