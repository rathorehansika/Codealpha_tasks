# Task 1 - Credit Scoring Model

Predict an individual's creditworthiness using past financial data. Built as part of the CodeAlpha ML internship.

## Objective
Predict whether an individual is a good or bad credit risk based on their financial history.

## Approach
- Performed feature engineering on financial history data (checking status, credit amount, duration, savings status, etc.)
- Trained and compared classification algorithms including Logistic Regression, Decision Trees, and Random Forest
- Evaluated models using accuracy, Precision, Recall, F1-Score, and ROC-AUC
- Analyzed feature importance to identify key predictors of credit risk

## Model
- **Best model**: Random Forest (`credit_scoring_model.joblib`)
- **Top predictive features**: checking status, credit amount, duration, age, savings status

## Project Structure
task-1-CreditScore/
├── plots/                            # Feature importance & evaluation charts
├── Credit_Score_Classification.ipynb # Main notebook (EDA, training, evaluation)
├── train_model.py                    # Model training script
├── credit_scoring_model.joblib       # Saved trained model
└── README.md
## Dataset
Dataset used: German Credit Data (Statlog)

