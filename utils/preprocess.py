"""
Data preprocessing utilities for Credit Card Approval Prediction.
Handles cleaning, encoding, scaling, outlier treatment, and feature engineering.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Column definitions matching the dataset and web form
FEATURE_COLUMNS = [
    "Gender",
    "Age",
    "Annual_Income",
    "Employment_Status",
    "Years_of_Employment",
    "Education_Level",
    "Marital_Status",
    "Number_of_Dependents",
    "Existing_Loan_Amount",
    "Monthly_Income",
    "Credit_Score",
    "Loan_Status",
    "Credit_History",
    "Number_of_Credit_Inquiries",
    "Debt_to_Income_Ratio",
]

CATEGORICAL_COLUMNS = [
    "Gender",
    "Employment_Status",
    "Education_Level",
    "Marital_Status",
    "Loan_Status",
    "Credit_History",
]

NUMERIC_COLUMNS = [
    "Age",
    "Annual_Income",
    "Years_of_Employment",
    "Number_of_Dependents",
    "Existing_Loan_Amount",
    "Monthly_Income",
    "Credit_Score",
    "Number_of_Credit_Inquiries",
    "Debt_to_Income_Ratio",
]

TARGET_COLUMN = "Approval_Status"


class DataPreprocessor:
    """Encapsulates all preprocessing steps for training and inference."""

    def __init__(self):
        self.label_encoders = {}
        self.target_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_columns = FEATURE_COLUMNS.copy()
        self.is_fitted = False

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows from the dataset."""
        before = len(df)
        df = df.drop_duplicates()
        removed = before - len(df)
        if removed > 0:
            print(f"Removed {removed} duplicate rows.")
        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values using appropriate strategies per column type."""
        df = df.copy()
        for col in NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        for col in CATEGORICAL_COLUMNS:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0])
        if TARGET_COLUMN in df.columns:
            df = df.dropna(subset=[TARGET_COLUMN])
        return df

    def treat_outliers(self, df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
        """Cap outliers using IQR method for numeric columns."""
        df = df.copy()
        cols = columns or NUMERIC_COLUMNS
        for col in cols:
            if col not in df.columns:
                continue
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            df[col] = df[col].clip(lower=lower, upper=upper)
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features from existing columns."""
        df = df.copy()
        if "Annual_Income" in df.columns and "Monthly_Income" in df.columns:
            df["Income_Consistency"] = df["Monthly_Income"] / (
                df["Annual_Income"] / 12 + 1
            )
        if "Existing_Loan_Amount" in df.columns and "Annual_Income" in df.columns:
            df["Loan_to_Income_Ratio"] = df["Existing_Loan_Amount"] / (
                df["Annual_Income"] + 1
            )
        if "Credit_Score" in df.columns and "Number_of_Credit_Inquiries" in df.columns:
            df["Credit_Risk_Score"] = df["Credit_Score"] - (
                df["Number_of_Credit_Inquiries"] * 5
            )
        if "Age" in df.columns and "Years_of_Employment" in df.columns:
            df["Employment_Ratio"] = df["Years_of_Employment"] / (df["Age"] + 1)
        return df

    def fit_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit label encoders on categorical columns."""
        df = df.copy()
        for col in CATEGORICAL_COLUMNS:
            if col not in df.columns:
                continue
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
        if TARGET_COLUMN in df.columns:
            # Approved=1, Rejected=0 for consistent ROC/probability interpretation
            self.target_encoder.fit(["Rejected", "Approved"])
            df[TARGET_COLUMN] = self.target_encoder.transform(
                df[TARGET_COLUMN].astype(str)
            )
        return df

    def transform_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted label encoders to categorical columns."""
        df = df.copy()
        for col in CATEGORICAL_COLUMNS:
            if col not in df.columns or col not in self.label_encoders:
                continue
            le = self.label_encoders[col]
            values = df[col].astype(str)
            # Handle unseen categories by mapping to most frequent class
            known = set(le.classes_)
            values = values.apply(lambda x: x if x in known else le.classes_[0])
            df[col] = le.transform(values)
        return df

    def get_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select feature columns including engineered ones."""
        engineered = [
            "Income_Consistency",
            "Loan_to_Income_Ratio",
            "Credit_Risk_Score",
            "Employment_Ratio",
        ]
        all_features = self.feature_columns + [
            c for c in engineered if c in df.columns
        ]
        return df[all_features]

    def fit(self, df: pd.DataFrame) -> tuple:
        """
        Full preprocessing pipeline for training.
        Returns (X_train, X_test, y_train, y_test).
        """
        df = self.remove_duplicates(df)
        df = self.handle_missing_values(df)
        df = self.treat_outliers(df)
        df = self.engineer_features(df)
        df = self.fit_encode(df)

        X = self.get_feature_matrix(df)
        y = df[TARGET_COLUMN]

        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        self.is_fitted = True
        return X_train, X_test, y_train, y_test

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess a single row or batch for prediction."""
        df = self.handle_missing_values(df)
        df = self.treat_outliers(df)
        df = self.engineer_features(df)
        df = self.transform_encode(df)
        X = self.get_feature_matrix(df)
        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns)

    def decode_target(self, encoded_value: int) -> str:
        """Convert encoded prediction back to label string."""
        return self.target_encoder.inverse_transform([encoded_value])[0]

    def encode_input(self, data: dict) -> pd.DataFrame:
        """Convert form input dict to DataFrame with correct column names."""
        row = {
            "Gender": data.get("gender", "Male"),
            "Age": float(data.get("age", 30)),
            "Annual_Income": float(data.get("annual_income", 50000)),
            "Employment_Status": data.get("employment_status", "Employed"),
            "Years_of_Employment": float(data.get("years_of_employment", 5)),
            "Education_Level": data.get("education_level", "Bachelor"),
            "Marital_Status": data.get("marital_status", "Single"),
            "Number_of_Dependents": float(data.get("number_of_dependents", 0)),
            "Existing_Loan_Amount": float(data.get("existing_loan_amount", 0)),
            "Monthly_Income": float(data.get("monthly_income", 4000)),
            "Credit_Score": float(data.get("credit_score", 650)),
            "Loan_Status": data.get("loan_status", "None"),
            "Credit_History": data.get("credit_history", "Good"),
            "Number_of_Credit_Inquiries": float(
                data.get("number_of_credit_inquiries", 1)
            ),
            "Debt_to_Income_Ratio": float(data.get("debt_to_income_ratio", 0.3)),
        }
        return pd.DataFrame([row])
