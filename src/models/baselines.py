"""
Phase 4: Classical Machine Learning Baselines
--------------------------------------------
Before building our PyTorch Deep Learning model, we must establish benchmarks
using classical ML models: Logistic Regression & Random Forest.

Key Evaluation Concepts:
- Accuracy alone is misleading due to class imbalance!
- Recall is critical in Cybersecurity: Missing an attack (False Negative) is far more dangerous
  than a false alarm (False Positive).
- Macro F1 treats all attack classes equally regardless of frequency.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")

def load_processed_data():
    data_path = os.path.join(PROCESSED_DIR, "processed_data.npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError("Processed dataset not found! Run preprocess.py first.")
    
    data = np.load(data_path)
    return data['X_train'], data['y_train'], data['X_test'], data['y_test']

def evaluate_model(name: str, model, X_test, y_test):
    print(f"\n=================== EVALUATING {name.upper()} ===================")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)

    print(f"Accuracy:          {acc:.4f}")
    print(f"Macro F1-Score:    {f1_macro:.4f}")
    print(f"Weighted F1-Score: {f1_weighted:.4f}")
    print(f"Macro Recall:      {r_macro:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro)
    }
    return metrics

def train_baselines():
    X_train, y_train, X_test, y_test = load_processed_data()

    results = {}

    # 1. Logistic Regression Baseline
    print("[INFO] Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    results["Logistic Regression"] = evaluate_model("Logistic Regression", lr, X_test, y_test)

    # 2. Random Forest Baseline
    print("\n[INFO] Training Random Forest Classifier (100 trees)...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    results["Random Forest"] = evaluate_model("Random Forest", rf, X_test, y_test)

    # Save results to json
    results_path = os.path.join(MODELS_DIR, "baseline_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\n[SUCCESS] Baseline evaluation results saved to {results_path}")

if __name__ == "__main__":
    train_baselines()
