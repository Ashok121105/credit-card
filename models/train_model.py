"""
Train and evaluate multiple ML models for Credit Card Approval Prediction.
Run: python models/train_model.py
"""

import os
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from utils.preprocess import DataPreprocessor
from utils.helper import (
    DATASET_PATH,
    evaluate_model,
    get_dataset_overview,
    plot_accuracy_comparison,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_model_performance,
    plot_roc_curve,
    save_metrics_json,
    save_model_artifact,
    select_best_model,
)


def generate_sample_dataset(path: str, n_samples: int = 1500):
    """Generate a realistic sample credit card approval dataset."""
    import numpy as np

    np.random.seed(42)
    genders = ["Male", "Female"]
    employment = ["Employed", "Self-Employed", "Unemployed", "Retired"]
    education = ["High School", "Bachelor", "Master", "PhD", "Other"]
    marital = ["Single", "Married", "Divorced"]
    loan_status = ["None", "Active", "Closed", "Default"]
    credit_history = ["Good", "Average", "Poor", "No History"]

    data = []
    for _ in range(n_samples):
        age = int(np.random.randint(21, 65))
        annual_income = float(np.random.uniform(20000, 150000))
        monthly_income = annual_income / 12 * np.random.uniform(0.85, 1.0)
        credit_score = int(np.clip(np.random.normal(650, 80), 300, 850))
        years_emp = float(np.clip(np.random.exponential(5), 0, age - 18))
        dependents = int(np.random.randint(0, 5))
        existing_loan = float(np.random.uniform(0, annual_income * 0.5))
        inquiries = int(np.random.randint(0, 8))
        dti = float(np.clip(existing_loan / (annual_income + 1) + np.random.uniform(0, 0.3), 0, 1))

        gender = np.random.choice(genders)
        emp_status = np.random.choice(employment, p=[0.65, 0.15, 0.1, 0.1])
        edu = np.random.choice(education, p=[0.2, 0.4, 0.25, 0.1, 0.05])
        mar = np.random.choice(marital, p=[0.4, 0.45, 0.15])
        loan_st = np.random.choice(loan_status, p=[0.5, 0.25, 0.15, 0.1])
        credit_hist = np.random.choice(credit_history, p=[0.5, 0.25, 0.15, 0.1])

        # Approval logic based on financial health
        score = (
            (credit_score / 850) * 40
            + (1 - dti) * 25
            + min(years_emp / 20, 1) * 15
            + (1 if credit_hist == "Good" else 0.3) * 10
            + (1 if emp_status == "Employed" else 0.2) * 10
            - inquiries * 2
        )
        approval = "Approved" if score > 45 and np.random.random() > 0.15 else "Rejected"

        data.append({
            "Gender": gender,
            "Age": age,
            "Annual_Income": round(annual_income, 2),
            "Employment_Status": emp_status,
            "Years_of_Employment": round(years_emp, 1),
            "Education_Level": edu,
            "Marital_Status": mar,
            "Number_of_Dependents": dependents,
            "Existing_Loan_Amount": round(existing_loan, 2),
            "Monthly_Income": round(monthly_income, 2),
            "Credit_Score": credit_score,
            "Loan_Status": loan_st,
            "Credit_History": credit_hist,
            "Number_of_Credit_Inquiries": inquiries,
            "Debt_to_Income_Ratio": round(dti, 4),
            "Approval_Status": approval,
        })

    df = pd.DataFrame(data)
    # Add some missing values and duplicates for preprocessing demo
    for col in ["Credit_Score", "Annual_Income", "Debt_to_Income_Ratio"]:
        idx = np.random.choice(len(df), size=20, replace=False)
        df.loc[idx, col] = np.nan
    df = pd.concat([df, df.sample(30, random_state=42)], ignore_index=True)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Sample dataset saved to {path} ({len(df)} records)")
    return df


def train_all_models():
    """Main training pipeline."""
    print("=" * 60)
    print("Credit Card Approval - Model Training Pipeline")
    print("=" * 60)

    if not os.path.exists(DATASET_PATH):
        df = generate_sample_dataset(DATASET_PATH)
    else:
        df = pd.read_csv(DATASET_PATH)
        print(f"Loaded dataset: {len(df)} records")

    dataset_info = get_dataset_overview(df)
    plot_class_distribution(df)

    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.fit(df)
    feature_names = list(X_train.columns)
    print(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
    print(f"Features: {len(feature_names)}")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=42, eval_metric="logloss",
        ),
    }

    all_metrics = []
    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, name)
        all_metrics.append(metrics)
        trained_models[name] = model

        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:  {metrics['roc_auc']:.4f}")

        plot_confusion_matrix(metrics["confusion_matrix"], name)
        y_proba = model.predict_proba(X_test)[:, 1]
        plot_roc_curve(y_test, y_proba, name)
        plot_feature_importance(model, feature_names, name)

    plot_accuracy_comparison(all_metrics)
    plot_model_performance(all_metrics)

    best = select_best_model(all_metrics)
    print(f"\n{'=' * 60}")
    print(f"Best Model: {best['model_name']} (F1: {best['f1_score']:.4f})")
    print(f"{'=' * 60}")

    # Comparison table
    comparison = pd.DataFrame(all_metrics)[
        ["model_name", "accuracy", "precision", "recall", "f1_score", "roc_auc"]
    ]
    print("\nModel Comparison Table:")
    print(comparison.to_string(index=False))

    best_model = trained_models[best["model_name"]]
    save_model_artifact(best_model, preprocessor, best, all_metrics, feature_names)
    save_metrics_json(all_metrics, dataset_info)

    print("\nTraining complete! Artifacts saved.")
    return best_model, preprocessor, all_metrics


if __name__ == "__main__":
    train_all_models()
