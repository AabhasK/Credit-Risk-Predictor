import sys

import gradio as gr
import joblib
import pandas as pd

bundle = joblib.load("model/loan_model.joblib")
model = bundle["model"]
defaults = bundle["defaults"]
options = bundle["options"]
threshold = 0.5


def predict(duration, amount, age, checking, history, savings, employment, purpose, housing):
    row = dict(defaults)
    row.update({
        "duration_months": duration,
        "credit_amount": amount,
        "age": age,
        "status_checking": checking,
        "credit_history": history,
        "savings": savings,
        "employment_since": employment,
        "purpose": purpose,
        "housing": housing,
    })
    applicant = pd.DataFrame([row])[bundle["columns"]]
    p = model.predict_proba(applicant)[0, 1]
    verdict = "High risk, review before approving" if p >= threshold else "Low risk"
    return {"Default": p, "No default": 1 - p}, verdict


def dropdown(col, label, value):
    return gr.Dropdown(options[col], value=value, label=label)


with gr.Blocks(title="Loan Default Predictor") as demo:
    gr.Markdown("# Loan Default Predictor\n"
                "Balanced Logistic Regression trained on the UCI German Credit dataset. "
                "Amounts are in Deutsche Mark.")
    with gr.Row():
        with gr.Column():
            duration = gr.Slider(4, 72, value=24, step=1, label="Loan duration (months)")
            amount = gr.Slider(250, 20000, value=3000, step=250, label="Credit amount")
            age = gr.Slider(19, 75, value=35, step=1, label="Age")
            checking = dropdown("status_checking", "Checking account", "< 0 DM")
            history = dropdown("credit_history", "Credit history", "existing paid till now")
            savings = dropdown("savings", "Savings", "< 100 DM")
            employment = dropdown("employment_since", "Employed for", "1 to 4 years")
            purpose = dropdown("purpose", "Purpose", "new car")
            housing = dropdown("housing", "Housing", "own")
            button = gr.Button("Predict", variant="primary")
        with gr.Column():
            probs = gr.Label(label="Predicted probability")
            verdict = gr.Textbox(label="Decision")

    inputs = [duration, amount, age, checking, history, savings, employment, purpose, housing]
    button.click(predict, inputs, [probs, verdict])

if __name__ == "__main__":
    demo.launch(share="--share" in sys.argv)
