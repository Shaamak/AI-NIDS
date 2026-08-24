"""
Unit Tests for PyTorch Neural Network Architecture (v2)
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
    model.eval()  # Required for BatchNorm with small batch sizes
    dummy_input = torch.randn(batch_size, input_dim)
    output = model(dummy_input)

    assert output.shape == (batch_size, num_classes), f"Expected shape ({batch_size}, {num_classes}), got {output.shape}"

def test_model_output_probabilities():
    """Test that softmax output sums to 1.0 for each sample."""
    model = NIDSNeuralNet(input_dim=100, num_classes=10)
    model.eval()
    dummy_input = torch.randn(8, 100)
    
    with torch.no_grad():
        logits = model(dummy_input)
        probs = torch.softmax(logits, dim=1)
    
    # Each row should sum to approximately 1.0
    row_sums = probs.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(8), atol=1e-5)

def test_dataset_len_and_getitem():
    X_dummy = torch.randn(50, 100).numpy()
    y_dummy = torch.randint(0, 10, (50,)).numpy()

    dataset = NIDSDataset(X_dummy, y_dummy)
    assert len(dataset) == 50

    x_sample, y_sample = dataset[0]
    assert x_sample.shape[0] == 100
    assert isinstance(y_sample, torch.Tensor)

def test_model_has_batchnorm():
    """Verify the upgraded model contains BatchNorm layers."""
    model = NIDSNeuralNet(input_dim=190, num_classes=10)
    bn_layers = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d)]
    assert len(bn_layers) >= 3, f"Expected at least 3 BatchNorm layers, found {len(bn_layers)}"
