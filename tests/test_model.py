"""Unit tests for PINN model architecture."""

import pytest
import torch
from src.models import PINNModel, physics_penalty


def test_pinn_initialization(sample_config):
    """Test PINN model initialization."""
    model = PINNModel(
        input_dim=sample_config.model.input_dim,
        hidden_dims=sample_config.model.hidden_dims
    )
    
    assert model.input_dim == 45
    assert model.output_dim == 1


def test_pinn_forward_pass(sample_config, sample_data):
    """Test PINN forward pass."""
    model = PINNModel(
        input_dim=sample_config.model.input_dim,
        hidden_dims=sample_config.model.hidden_dims
    )
    
    X, _ = sample_data
    output = model(X)
    
    assert output.shape == (X.shape[0], 1)
    assert not torch.isnan(output).any()


def test_pinn_output_activation():
    """Test that output activation is applied correctly."""
    model_relu = PINNModel(input_dim=10, output_activation="relu")
    model_softplus = PINNModel(input_dim=10, output_activation="softplus")
    
    X = torch.randn(5, 10)
    
    # ReLU output could be negative (before ReLU is applied)
    # Softplus output should always be positive
    output_softplus = model_softplus(X)
    assert torch.all(output_softplus >= 0), "Softplus should ensure positive output"


def test_pinn_forward_unnormalized(sample_data):
    """Test forward_unnormalized method."""
    model = PINNModel(input_dim=45)
    
    X, _ = sample_data
    output_norm = model(X)
    output_unnorm = model.forward_unnormalized(X)
    
    # Unnormalized should be different from normalized (due to activation)
    assert not torch.allclose(output_norm, output_unnorm)


def test_physics_penalty_violations():
    """Test physics penalty computation."""
    # Create predictions with violations
    pred_iv = torch.tensor([
        [0.5],   # Normal
        [-0.1],  # Below minimum (violation)
        [4.0]    # Above maximum (violation)
    ])
    
    penalty = physics_penalty(pred_iv, iv_min=0.0, iv_max=3.0)
    
    assert penalty.item() > 0, "Penalty should be positive for violations"
    assert not torch.isnan(penalty)


def test_physics_penalty_no_violations():
    """Test physics penalty for valid predictions."""
    pred_iv = torch.tensor([[0.5], [1.0], [1.5]])  # All valid
    
    penalty = physics_penalty(pred_iv, iv_min=0.0, iv_max=3.0)
    
    assert penalty.item() == 0, "Penalty should be zero for valid predictions"


def test_physics_penalty_flattening():
    """Test that physics penalty handles different tensor shapes."""
    # 1D tensor
    pred_1d = torch.tensor([0.5, 1.0, 1.5])
    penalty_1d = physics_penalty(pred_1d)
    
    # 2D tensor
    pred_2d = torch.tensor([[0.5], [1.0], [1.5]])
    penalty_2d = physics_penalty(pred_2d)
    
    assert torch.allclose(penalty_1d, penalty_2d)


def test_pinn_hidden_activation():
    """Test different hidden layer activations."""
    activations = ["relu", "tanh", "elu", "leaky_relu"]
    
    for activation in activations:
        model = PINNModel(input_dim=10, hidden_activation=activation)
        X = torch.randn(5, 10)
        output = model(X)
        
        assert output.shape == (5, 1)
        assert not torch.isnan(output).any()


def test_pinn_invalid_activation():
    """Test that invalid activation raises error."""
    with pytest.raises(ValueError):
        PINNModel(input_dim=10, hidden_activation="invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
