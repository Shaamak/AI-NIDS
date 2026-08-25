"""
Phase 8: Comprehensive Model Evaluation & Baseline Comparison
-------------------------------------------------------------
This script evaluates our trained PyTorch Multi-Layer Perceptron (MLP) on the untouched test set,
generates a detailed per-class classification report, plots a confusion matrix,
and constructs a side-by-side comparison table against our classical ML baselines.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix

from src.models.network import NIDSNeuralNet
from src.data.preprocess import NIDSPreprocessor

# Make NIDSPreprocessor accessible in __main__ namespace for joblib unpickling compatibility
import __main__
__main__.NIDSPreprocessor = NIDSPreprocessor

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "notebooks", "plots")

def evaluate_pytorch_model():
    print("=================== PHASE 8: FINAL MODEL EVALUATION ===================")
    
    # 1. Load test data
    data_path = os.path.join(PROCESSED_DIR, "processed_data.npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError("Processed dataset not found! Run preprocess.py first.")
    
    data = np.load(data_path)
    X_test, y_test = data['X_test'], data['y_test']

    # 2. Load preprocessor to get target class names
    preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.pkl")
    class_names = [f"Class {i}" for i in range(10)]
    if os.path.exists(preprocessor_path):
        preprocessor = joblib.load(preprocessor_path)
        class_names = list(preprocessor.label_encoder.classes_)

    # 3. Load model config & saved weights
    config_path = os.path.join(MODELS_DIR, "model_config.json")
    model_path = os.path.join(MODELS_DIR, "nids_model.pt")

    if not os.path.exists(config_path) or not os.path.exists(model_path):
        raise FileNotFoundError("PyTorch model checkpoints not found! Run train.py first.")

    with open(config_path, "r") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NIDSNeuralNet(input_dim=config["input_dim"], num_classes=config["num_classes"]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 4. Perform Inference on Test Set
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(X_test_tensor)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1).cpu().numpy()

    # 5. Calculate Multi-Class Metrics
    acc = accuracy_score(y_test, preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, preds, average='macro', zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test, preds, average='weighted', zero_division=0)

    # Calculate Binary Metrics (Normal vs Cyber Threat)
    # Find index of "Normal" class
    normal_idx = class_names.index("Normal") if "Normal" in class_names else 6
    y_test_binary = (y_test != normal_idx).astype(int)
    preds_binary = (preds != normal_idx).astype(int)
    
    bin_acc = accuracy_score(y_test_binary, preds_binary)
    bin_p, bin_r, bin_f1, _ = precision_recall_fscore_support(y_test_binary, preds_binary, average='binary', zero_division=0)

    print(f"\n--- PyTorch Neural Network Test Results ---")
    print(f"| Task                      | Accuracy | Precision | Recall | F1-Score |")
    print(f"|---------------------------|----------|-----------|--------|----------|")
    print(f"| Binary Threat Detection    | {bin_acc*100:.2f}%   | {bin_p*100:.2f}%    | {bin_r*100:.2f}% | {bin_f1*100:.2f}%   |")
    print(f"| 10-Class Threat Categorization | {acc*100:.2f}%   | {p_weighted*100:.2f}%    | {r_weighted*100:.2f}% | {f1_weighted*100:.2f}%   |")
    print(f"\nMacro F1-Score:    {f1_macro:.4f}")
    print(f"Macro Recall:      {r_macro:.4f}")
    
    print("\nDetailed Per-Class Classification Report:")
    report_str = classification_report(y_test, preds, target_names=class_names, zero_division=0)
    print(report_str)

    # 6. Plot Confusion Matrix
    os.makedirs(PLOTS_DIR, exist_ok=True)
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('PyTorch MLP Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    cm_path = os.path.join(PLOTS_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"[INFO] Saved Confusion Matrix plot to {cm_path}")

    # 7. Model Comparison Table
    baseline_path = os.path.join(MODELS_DIR, "baseline_results.json")
    comparison = {}

    if os.path.exists(baseline_path):
        with open(baseline_path, "r") as f:
            comparison = json.load(f)

    comparison["PyTorch MLP"] = {
        "accuracy": float(acc),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro)
    }

    df_comp = pd.DataFrame(comparison).T
    print("\n=================== MODEL COMPARISON SUMMARY ===================")
    print(df_comp[['accuracy', 'macro_recall', 'macro_f1', 'weighted_f1']].to_string())

    # Save comparison to file
    comp_save_path = os.path.join(MODELS_DIR, "final_comparison.json")
    with open(comp_save_path, "w") as f:
        json.dump(comparison, f, indent=4)
    print(f"\n[SUCCESS] Saved final evaluation comparison to {comp_save_path}")

if __name__ == "__main__":
    evaluate_pytorch_model()
