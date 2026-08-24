"""
Phase 5 (v2): Advanced PyTorch Neural Network Architecture & Dataset Loader
---------------------------------------------------------------------------
UPGRADES from v1:
  1. BatchNorm1d: Stabilizes training by normalizing hidden layer outputs.
  2. Wider Network: 256 -> 128 -> 64 -> 32 neurons (was 128 -> 64 -> 32).
  3. He Initialization: Optimal weight init for ReLU-based networks.
  4. Residual Skip Connection: Lets gradients flow more easily through deeper layers.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

# ==========================================
# 1. THE DATASET CLASS (unchanged)
# ==========================================
class NIDSDataset(Dataset):
    """
    Wraps Numpy arrays into PyTorch Tensors for batched data loading.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==========================================
# 2. ADVANCED NEURAL NETWORK ARCHITECTURE
# ==========================================
class NIDSNeuralNet(nn.Module):
    """
    Advanced Multi-Layer Perceptron with:
    - BatchNorm1d for training stability
    - He (Kaiming) weight initialization for ReLU networks
    - Wider hidden layers for greater representational capacity
    - Residual skip connection between layers 2 and 3
    """
    def __init__(self, input_dim: int, num_classes: int):
        super(NIDSNeuralNet, self).__init__()
        
        # ---- BLOCK 1: Input Projection ----
        # Projects raw features into a 256-dimensional hidden space.
        # BatchNorm1d normalizes the outputs across the batch to prevent
        # "internal covariate shift" (layer inputs drifting during training).
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # ---- BLOCK 2: Feature Extraction ----
        self.block2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # ---- BLOCK 3: Deep Feature Refinement ----
        self.block3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # ---- BLOCK 4: Fine-Grained Compression ----
        self.block4 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.15)
        )
        
        # ---- Residual Skip Connection ----
        # Projects block2 output (128-dim) down to 64-dim so it can be added
        # to block3 output. This creates a "shortcut" for gradient flow.
        self.skip_proj = nn.Linear(128, 64)
        
        # ---- Output Head ----
        self.output_layer = nn.Linear(32, num_classes)
        
        # ---- He (Kaiming) Weight Initialization ----
        # Standard Xavier init works well for sigmoid/tanh activations,
        # but ReLU "kills" half the values (negatives become 0).
        # He init compensates by using a larger variance: Var(w) = 2/n_in
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Block 1: Input -> 256
        x = self.block1(x)
        
        # Block 2: 256 -> 128
        x = self.block2(x)
        
        # Save block2 output for the skip connection
        residual = self.skip_proj(x)
        
        # Block 3: 128 -> 64
        x = self.block3(x)
        
        # Add the residual skip connection (element-wise addition)
        # This lets the network learn "corrections" on top of the skip path,
        # making optimization much easier for deeper networks.
        x = x + residual
        
        # Block 4: 64 -> 32
        x = self.block4(x)
        
        # Output: 32 -> num_classes (raw logits)
        out = self.output_layer(x)
        
        return out
