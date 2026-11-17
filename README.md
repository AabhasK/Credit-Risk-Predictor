# Loan Default Prediction

UML501 Machine Learning project by Aabhas Khandelwal (1024030673) and Joy Soni (1024030674), submitted to Mr. Deepak Sharma.

We predict loan default on the UCI German Credit dataset with Logistic Regression, Decision Tree and Random Forest, judged on recall for defaulters. Balanced class weights raise recall from 0.53 to 0.80 for Logistic Regression, which is the final model.

## Files

- `MLproject.ipynb` is the full notebook: cleaning, EDA, models, results, and it saves the model
- `app.py` is the Gradio demo
- `data/german.data` is the dataset
- `model/` holds the saved model and `figures/` the charts
- `report/` has the project report

## Run

```
pip install -r requirements.txt
jupyter notebook MLproject.ipynb
python app.py
```

Run all cells in the notebook first, then start the demo. It opens at http://127.0.0.1:7860.
