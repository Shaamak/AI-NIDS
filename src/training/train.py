"""
Phase 6: PyTorch Training Pipeline
----------------------------------
This script is the "engine" that trains our Deep Learning model.
It handles DataLoaders, the Forward Pass, the Backward Pass (Backpropagation),
and Model Checkpointing.
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
import numpy as np

# Import our custom classes from Phase 5
from src.models.network import NIDSDataset, NIDSNeuralNet

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")

def load_data():
    data_path = os.path.join(PROCESSED_DIR, "processed_data.npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError("Processed dataset not found! Run preprocess.py first.")
    
    data = np.load(data_path)
    return data['X_train'], data['y_train'], data['X_test'], data['y_test']

def train_model():
    print("[INFO] Loading preprocessed data...")
    X_train, y_train, X_test, y_test = load_data()

    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    print(f"[INFO] Features: {input_dim}, Classes: {num_classes}")

    # 1. DATASETS & DATALOADERS
    # We wrap our Numpy arrays in PyTorch Datasets
    train_dataset = NIDSDataset(X_train, y_train)
    test_dataset = NIDSDataset(X_test, y_test)

    # DataLoader handles shuffling and grouping into batches (e.g., 64 rows at a time)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 2. DEVICE & MODEL INITIALIZATION
    # This makes the code device-independent (runs on GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on device: {device}")

    model = NIDSNeuralNet(input_dim=input_dim, num_classes=num_classes).to(device)

    # 3. LOSS FUNCTION & OPTIMIZER
    # CrossEntropyLoss automatically applies Softmax and calculates how wrong our predictions are
    criterion = nn.CrossEntropyLoss()
    
    # Adam is our optimizer. It updates the network weights to minimize the loss.
    # lr=0.001 is the "learning rate" (how big of a step it takes when adjusting weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. TRAINING LOOP
    epochs = 20
    best_val_loss = float('inf')
    model_save_path = os.path.join(MODELS_DIR, "nids_model.pt")
    config_save_path = os.path.join(MODELS_DIR, "model_config.json")

    print("\nStarting Training...\n")
    for epoch in range(epochs):
        # Put model in "Training Mode" (Enables Dropout)
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # --- THE MAGIC 3 STEPS OF BACKPROPAGATION ---
            
            # Step 1: Zero out old gradients (flush the memory)
            optimizer.zero_grad()
            
            # Step 2: Forward Pass (Make predictions)
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            
            # Step 3: Backward Pass & Step (Calculate errors & update weights)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # 5. VALIDATION LOOP
        # Put model in "Evaluation Mode" (Disables Dropout for fair testing)
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        # torch.no_grad() tells PyTorch to stop tracking gradients to save memory
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                predictions = model(batch_X)
                
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()

                # Get the predicted class (the index with the highest score)
                _, predicted = torch.max(predictions.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        val_accuracy = 100 * correct / total

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

        # 6. MODEL CHECKPOINTING (Save the best model)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            # Save configuration so FastAPI knows how to rebuild the model later
            with open(config_save_path, "w") as f:
                json.dump({"input_dim": input_dim, "num_classes": num_classes}, f)
            print(f"   -> Validation loss improved! Saved model to {model_save_path}")

    print("\n[SUCCESS] Training Complete.")

if __name__ == "__main__":
    train_model()
