# Loan Default Prediction

UML501 Machine Learning project by Aabhas Khandelwal (1024030673) and Joy Soni (1024030674), submitted to Mr. Deepak Sharma.

We predict loan default on the UCI German Credit dataset with Logistic Regression, Decision Tree, Random Forest, XGBoost and LightGBM, judged on recall for defaulters. Balanced class weights raise recall from 0.53 to 0.80 for Logistic Regression. Hyperparameter tuning and ROC-AUC confirm it as the final model.

## Files

- `MLproject.ipynb` is the full notebook: cleaning, EDA, models and results
- `data/german.data` is the dataset
- `figures/` holds the charts
- `report/` has the project report

## Run

```
pip install -r requirements.txt
jupyter notebook MLproject.ipynb
```

Run all cells in order.

## Demo

```
python app.py
```

Open http://127.0.0.1:7860. The app trains the same five balanced models on the same split as the notebook, then scores an applicant you enter in the form.
