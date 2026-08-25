"""
Phase 6 (v2): Advanced PyTorch Training Pipeline
-------------------------------------------------
UPGRADES from v1:
  1. Class-Weighted Loss: Penalizes the model more for missing rare attack classes.
  2. Learning Rate Scheduler (ReduceLROnPlateau): Auto-reduces LR when loss stalls.
  3. Early Stopping: Stops training if validation loss hasn't improved in N epochs.
  4. Larger batch size (256) for more stable gradient estimates with BatchNorm.
  5. More epochs (50) with early stopping safety net.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import Counter
import numpy as np

from src.models.network import NIDSDataset, NIDSNeuralNet

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

def load_data():
    data_path = os.path.join(PROCESSED_DIR, "processed_data.npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError("Processed dataset not found! Run preprocess.py first.")
    
    data = np.load(data_path)
    return data['X_train'], data['y_train'], data['X_test'], data['y_test']

def compute_class_weights(y_train, num_classes, device):
    """
    Smoothed Square-Root Class Weighting:
    Formula: weight_i = sqrt( total_samples / (num_classes * count_of_class_i) )
    
    WHY?
    Raw inverse weighting (135x for Worms) is too aggressive and ruins Normal precision.
    Square-root smoothing scales the penalty down (135x -> 11.6x), balancing total
    accuracy AND minority attack recall!
    """
    class_counts = Counter(y_train)
    total = len(y_train)
    weights = []
    for i in range(num_classes):
        count = class_counts.get(i, 1)
        raw_w = total / (num_classes * count)
        smoothed_w = np.sqrt(raw_w) # Square root smoothing
        weights.append(smoothed_w)
    
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"[INFO] Smoothed Class Weights: {[f'{w:.2f}' for w in weights]}")
    return weights_tensor

def train_model():
    print("=" * 60)
    print("  AI-NIDS Advanced Training Pipeline v2")
    print("=" * 60)
    
    print("\n[INFO] Loading preprocessed data...")
    X_train, y_train, X_test, y_test = load_data()

    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    print(f"[INFO] Features: {input_dim}, Classes: {num_classes}")
    print(f"[INFO] Train samples: {len(X_train):,}, Test samples: {len(X_test):,}")

    # 1. DATASETS & DATALOADERS
    train_dataset = NIDSDataset(X_train, y_train)
    test_dataset = NIDSDataset(X_test, y_test)

    # Larger batch size (256) works better with BatchNorm because 
    # the batch statistics (mean/variance) are more representative.
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # 2. DEVICE & MODEL
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on device: {device}")

    model = NIDSNeuralNet(input_dim=input_dim, num_classes=num_classes).to(device)
    
    # Print model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # 3. CLASS-WEIGHTED LOSS FUNCTION
    # This is the single biggest accuracy booster for imbalanced datasets.
    class_weights = compute_class_weights(y_train, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 4. OPTIMIZER WITH WEIGHT DECAY (L2 Regularization)
    # weight_decay adds a penalty for large weights, preventing overfitting.
    # It is equivalent to L2 regularization: Loss = CE_Loss + lambda * ||W||^2
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # 5. LEARNING RATE SCHEDULER
    # ReduceLROnPlateau watches the validation loss.
    # If it hasn't improved for 'patience' epochs, it reduces lr by factor 0.5.
    # This lets the model make big jumps early and fine-tune later.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # 6. TRAINING LOOP WITH EARLY STOPPING
    epochs = 50
    best_val_loss = float('inf')
    patience = 7         # Stop if no improvement for 7 consecutive epochs
    patience_counter = 0
    model_save_path = os.path.join(MODELS_DIR, "nids_model.pt")
    config_save_path = os.path.join(MODELS_DIR, "model_config.json")

    print(f"\n{'='*60}")
    print(f"  Starting Training: {epochs} epochs, batch_size=256")
    print(f"  Early stopping patience: {patience}")
    print(f"{'='*60}\n")

    for epoch in range(epochs):
        # ---- TRAINING PHASE ----
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(predictions.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()

        # ---- VALIDATION PHASE ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                predictions = model(batch_X)
                
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()

                _, predicted = torch.max(predictions.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        train_accuracy = 100 * train_correct / train_total
        val_accuracy = 100 * val_correct / val_total
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch+1:02d}/{epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}% | "
              f"LR: {current_lr:.6f}")

        # Step the scheduler (watches validation loss)
        scheduler.step(avg_val_loss)

        # ---- MODEL CHECKPOINTING ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            with open(config_save_path, "w") as f:
                json.dump({"input_dim": input_dim, "num_classes": num_classes}, f)
            print(f"   -> Best model saved! (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[EARLY STOPPING] No improvement for {patience} consecutive epochs.")
                print(f"[EARLY STOPPING] Best Val Loss: {best_val_loss:.4f}")
                break

    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"  Best Validation Loss: {best_val_loss:.4f}")
    print(f"  Model saved to: {model_save_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    train_model()
