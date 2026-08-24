"""
Unit Tests for PyTorch Neural Network Architecture
"""

import sys
import os
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.network import NIDSNeuralNet, NIDSDataset

def test_model_forward_pass_shape():
    input_dim = 190
    num_classes = 10
    batch_size = 32

    model = NIDSNeuralNet(input_dim=input_dim, num_classes=num_classes)
    dummy_input = torch.randn(batch_size, input_dim)
    output = model(dummy_input)

    assert output.shape == (batch_size, num_classes), f"Expected shape ({batch_size}, {num_classes}), got {output.shape}"

def test_dataset_len_and_getitem():
    X_dummy = torch.randn(50, 100).numpy()
    y_dummy = torch.randint(0, 10, (50,)).numpy()

    dataset = NIDSDataset(X_dummy, y_dummy)
    assert len(dataset) == 50

    x_sample, y_sample = dataset[0]
    assert x_sample.shape[0] == 100
    assert isinstance(y_sample, torch.Tensor)
