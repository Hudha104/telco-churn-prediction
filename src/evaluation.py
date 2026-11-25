"""Shared evaluation helpers for the churn models (CM2604, Part 01)."""
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)


def evaluate(model, X, y, name: str, setting: str) -> dict:
    """Return the evaluation metrics for the positive class (Churn = 1)."""
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "Model": name,
        "Setting": setting,
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred),
        "Recall": recall_score(y, pred),
        "F1": f1_score(y, pred),
        "ROC-AUC": roc_auc_score(y, proba),
    }


def results_table(rows: list) -> pd.DataFrame:
    return pd.DataFrame(rows).round(4)
