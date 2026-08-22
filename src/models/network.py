"""
Phase 5: PyTorch Neural Network Architecture & Dataset Loader
-------------------------------------------------------------
This file contains the core PyTorch components for our Deep Learning model.

1. NIDSDataset: Wraps our numpy arrays into PyTorch Tensors.
2. NIDSNeuralNet: The actual Neural Network architecture (Multi-Layer Perceptron).
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

# ==========================================
# 1. THE DATASET CLASS
# ==========================================
class NIDSDataset(Dataset):
    """
    WHAT IT DOES:
    PyTorch models cannot read raw Pandas DataFrames or Numpy arrays directly during training in a batched way efficiently.
    They need a `Dataset` object that tells PyTorch:
    1. How many total samples there are (using __len__).
    2. How to grab one specific sample and its label at a given index (using __getitem__).

    WHY WE NEED IT:
    This allows PyTorch's `DataLoader` (which we will build in Phase 6) to automatically
    shuffle the data and group it into "batches" (e.g., 64 rows at a time).
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # Convert Numpy arrays to PyTorch Tensors.
        # X is converted to float32 (standard for neural network weights).
        # y is converted to int64 (standard for multi-class classification labels).
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        # Returns the total number of rows in the dataset
        return len(self.X)

    def __getitem__(self, idx):
        # Grabs the features and the label for a single row at 'idx'
        return self.X[idx], self.y[idx]


# ==========================================
# 2. THE NEURAL NETWORK ARCHITECTURE
# ==========================================
class NIDSNeuralNet(nn.Module):
    """
    WHAT IT DOES:
    This defines a Multi-Layer Perceptron (MLP) - a standard feed-forward neural network.
    
    HOW IT WORKS:
    - nn.Module is the base class for ALL neural networks in PyTorch. 
    - In __init__, we define the "layers" (the building blocks).
    - In forward(), we define the "path" the data takes through those layers.
    """
    def __init__(self, input_dim: int, num_classes: int):
        super(NIDSNeuralNet, self).__init__()
        
        # Linear Layers (Dense Layers): 
        # These perform the math: Output = (Input * Weights) + Bias.
        # It connects every input feature to a set of hidden "neurons".
        
        # Dropout Layers:
        # Randomly turns off a percentage of neurons during training (e.g., 30%).
        # WHY? It forces the network to learn robust patterns instead of relying on a single feature.
        # This prevents "overfitting" (memorizing the training data).

        self.layer1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(0.3)
        
        self.layer2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        
        self.layer3 = nn.Linear(64, 32)
        self.dropout3 = nn.Dropout(0.2)
        
        # The final output layer must output exactly `num_classes` values.
        # For us, it will output 10 values (the raw confidence scores for each of the 10 categories).
        self.output_layer = nn.Linear(32, num_classes)
        
        # Activation Function (ReLU):
        # Rectified Linear Unit. Math: max(0, x). 
        # If a neuron's value is negative, it makes it 0. If positive, it leaves it alone.
        # WHY? Without activation functions, no matter how many layers we stack, 
        # the network would just be one giant linear equation. ReLU allows it to learn complex, non-linear patterns.
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        WHAT IT DOES:
        This defines how the data 'x' flows through the network from start to finish.
        Whenever you pass data to the model like `predictions = model(data)`, it runs this function.
        """
        # Pass input through Layer 1, then apply ReLU, then Dropout
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        # Pass through Layer 2
        x = self.layer2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        # Pass through Layer 3
        x = self.layer3(x)
        x = self.relu(x)
        x = self.dropout3(x)
        
        # Final Output Layer (No ReLU or Dropout here!)
        # We return the raw, un-normalized scores (called "logits").
        # The loss function (CrossEntropyLoss) we use later will automatically apply Softmax to these logits.
        out = self.output_layer(x)
        
        return out
