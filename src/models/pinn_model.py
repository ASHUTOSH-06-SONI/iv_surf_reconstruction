
"""Physics-Informed Neural Network (PINN) for IV surface reconstruction."""

import torch
import torch.nn as nn
from typing import Optional, List


class PINNModel(nn.Module):
    """Physics-Informed Neural Network for implied volatility prediction.
    
    Architecture:
        - Multi-layer feedforward network with configurable hidden dimensions
        - Output activation: Softplus ensures positive IV predictions
        - Physics constraints: Penalizes IV violations (IV < 0 or IV > 3.0)
    
    Physics Constraints:
        - IV must be positive (enforced by Softplus activation)
        - IV typically ranges [0, 3.0] for NIFTY50 options
        - Soft constraint: L2 penalty on violations rather than hard clipping
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 64],
        output_dim: int = 1,
        hidden_activation: str = "relu",
        output_activation: str = "softplus",
        dropout_rate: float = 0.0,
    ):
        """Initialize PINN model.
        
        Args:
            input_dim: Number of input features (option characteristics + market features)
            hidden_dims: List of hidden layer dimensions. Default: [64, 64]
            output_dim: Output dimension (typically 1 for scalar IV prediction). Default: 1
            hidden_activation: Activation function for hidden layers. Default: "relu"
            output_activation: Activation function for output layer. Default: "softplus"
            dropout_rate: Dropout probability. Default: 0.0 (no dropout)
        
        Examples:
            >>> model = PINNModel(input_dim=45, hidden_dims=[128, 64])
            >>> x = torch.randn(32, 45)  # batch_size=32, features=45
            >>> y = model(x)  # shape: (32, 1)
        """
        super(PINNModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.hidden_activation = hidden_activation.lower()
        self.output_activation = output_activation.lower()
        self.dropout_rate = dropout_rate
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Add activation function
            if self.hidden_activation == "relu":
                layers.append(nn.ReLU())
            elif self.hidden_activation == "tanh":
                layers.append(nn.Tanh())
            elif self.hidden_activation == "elu":
                layers.append(nn.ELU())
            elif self.hidden_activation == "leaky_relu":
                layers.append(nn.LeakyReLU())
            else:
                raise ValueError(f"Unknown activation: {self.hidden_activation}")
            
            # Add dropout if specified
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Output layer (no activation here, will add after)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # Output activation (separate to allow unnormalized predictions when needed)
        if self.output_activation == "softplus":
            self.output_act = nn.Softplus()
        elif self.output_activation == "relu":
            self.output_act = nn.ReLU()
        elif self.output_activation == "sigmoid":
            self.output_act = nn.Sigmoid()
        elif self.output_activation == "none":
            self.output_act = nn.Identity()
        else:
            raise ValueError(f"Unknown output activation: {self.output_activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through PINN.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Predicted IV values of shape (batch_size, output_dim)
        """
        x = self.hidden_layers(x)
        x = self.output_act(x)
        return x
    
    def forward_unnormalized(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning unnormalized logits (before output activation).
        
        Useful for computing physics penalties on raw outputs.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Unnormalized output of shape (batch_size, output_dim)
        """
        return self.hidden_layers(x)


def physics_penalty(pred_sigma: torch.Tensor, iv_min: float = 0.0, iv_max: float = 3.0) -> torch.Tensor:
    """Compute physics constraint penalty for IV predictions.
    
    Penalizes predictions that violate financial constraints:
    - IV < iv_min (typically 0, negative volatility is unphysical)
    - IV > iv_max (typically 3.0 for NIFTY50 options)
    
    This is a soft constraint (L2 penalty) that allows flexibility during training
    while encouraging physically realistic predictions.
    
    Args:
        pred_sigma: Predicted IV values, shape (batch_size,) or (batch_size, 1)
        iv_min: Minimum allowed IV. Default: 0.0
        iv_max: Maximum allowed IV. Default: 3.0
    
    Returns:
        Scalar loss value (mean squared violations)
    
    Examples:
        >>> pred_iv = torch.tensor([[0.5], [-0.1], [4.0]])
        >>> penalty = physics_penalty(pred_iv, iv_min=0.0, iv_max=3.0)
        >>> # Penalizes -0.1 (< 0) and 4.0 (> 3.0)
    """
    # Flatten if needed
    if pred_sigma.dim() > 1:
        pred_sigma = pred_sigma.squeeze()
    
    # Lower bound violation: IV < iv_min
    lower_violation = torch.clamp(iv_min - pred_sigma, min=0)
    lower_penalty = torch.mean(lower_violation ** 2)
    
    # Upper bound violation: IV > iv_max
    upper_violation = torch.clamp(pred_sigma - iv_max, min=0)
    upper_penalty = torch.mean(upper_violation ** 2)
    
    return lower_penalty + upper_penalty
