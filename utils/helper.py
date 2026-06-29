"""
Helper utilities for model evaluation, visualization, and persistence.
"""

import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_IMAGES_DIR = os.path.join(BASE_DIR, "static", "images")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "static", "images", "metrics.json")
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "credit_card.csv")


def ensure_dirs():
    os.makedirs(STATIC_IMAGES_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "dataset"), exist_ok=True)


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    y_pred = model.predict(X_test)
    y_proba = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else y_pred
    )
    metrics = {
        "model_name": model_name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        ),
    }
    return metrics


def plot_class_distribution(df, target_col="Approval_Status", save_path=None):
    ensure_dirs()
    save_path = save_path or os.path.join(STATIC_IMAGES_DIR, "class_distribution.png")
    counts = df[target_col].value_counts()
    plt.figure(figsize=(8, 5))
    colors = ["#28a745", "#dc3545"]
    bars = plt.bar(counts.index, counts.values, color=colors[:len(counts)])
    plt.title("Class Distribution - Approval Status", fontsize=14, fontweight="bold")
    plt.xlabel("Approval Status")
    plt.ylabel("Count")
    for bar, val in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 str(val), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    return save_path


def plot_accuracy_comparison(metrics_list, save_path=None):
    ensure_dirs()
    save_path = save_path or os.path.join(STATIC_IMAGES_DIR, "accuracy_comparison.png")
    names = [m["model_name"] for m in metrics_list]
    accuracies = [m["accuracy"] for m in metrics_list]
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
    bars = plt.bar(names, accuracies, color=colors)
    plt.title("Model Accuracy Comparison", fontsize=14, fontweight="bold")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    for bar, val in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.4f}", ha="center", fontsize=10)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    return save_path


def plot_model_performance(metrics_list, save_path=None):
    ensure_dirs()
    save_path = save_path or os.path.join(STATIC_IMAGES_DIR, "model_performance.png")
    metrics_names = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    x = np.arange(len(metrics_names))
    width = 0.15
    plt.figure(figsize=(12, 6))
    for i, m in enumerate(metrics_list):
        values = [m[k] for k in metrics_names]
        plt.bar(x + i * width, values, width, label=m["model_name"])
    plt.xlabel("Metrics")
    plt.ylabel("Score")
    plt.title("Model Performance Comparison", fontsize=14, fontweight="bold")
    plt.xticks(x + width * (len(metrics_list) - 1) / 2, metrics_names)
    plt.legend()
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    return save_path


def plot_confusion_matrix(cm, model_name, save_path=None):
    ensure_dirs()
    if save_path is None:
        safe_name = model_name.replace(" ", "_").lower()
        save_path = os.path.join(STATIC_IMAGES_DIR, f"confusion_matrix_{safe_name}.png")
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Rejected", "Approved"],
                yticklabels=["Rejected", "Approved"])
    plt.title(f"Confusion Matrix - {model_name}", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    return save_path


def plot_roc_curve(y_test, y_proba, model_name, save_path=None):
    ensure_dirs()
    if save_path is None:
        safe_name = model_name.replace(" ", "_").lower()
        save_path = os.path.join(STATIC_IMAGES_DIR, f"roc_curve_{safe_name}.png")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#0d6efd", lw=2, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    return save_path


def plot_feature_importance(model, feature_names, model_name, save_path=None):
    ensure_dirs()
    if save_path is None:
        safe_name = model_name.replace(" ", "_").lower()
        save_path = os.path.join(STATIC_IMAGES_DIR, f"feature_importance_{safe_name}.png")
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return None
    indices = np.argsort(importances)[::-1][:15]
    plt.figure(figsize=(10, 6))
    plt.barh([feature_names[i] for i in indices[::-1]], importances[indices[::-1]], color="#0d6efd")
    plt.title(f"Feature Importance - {model_name}", fontsize=13, fontweight="bold")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    return save_path


def select_best_model(metrics_list) -> dict:
    return max(metrics_list, key=lambda m: m["f1_score"])


def save_model_artifact(model, preprocessor, best_metrics, all_metrics, feature_names):
    ensure_dirs()
    artifact = {
        "model": model,
        "preprocessor": preprocessor,
        "best_model_name": best_metrics["model_name"],
        "metrics": all_metrics,
        "best_metrics": best_metrics,
        "feature_names": list(feature_names),
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)
    print(f"Model saved to {MODEL_PATH}")


def load_model_artifact():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def save_metrics_json(metrics_list, dataset_info=None):
    ensure_dirs()
    data = {
        "models": metrics_list,
        "best_model": select_best_model(metrics_list),
        "dataset_info": dataset_info or {},
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(data, f, indent=2)
    return METRICS_PATH


def load_metrics_json():
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH, "r") as f:
        return json.load(f)


def get_dataset_overview(df) -> dict:
    return {
        "total_records": len(df),
        "total_features": len(df.columns) - 1,
        "approved_count": int((df["Approval_Status"] == "Approved").sum()),
        "rejected_count": int((df["Approval_Status"] == "Rejected").sum()),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_count": int(df.duplicated().sum()),
        "columns": list(df.columns),
    }
