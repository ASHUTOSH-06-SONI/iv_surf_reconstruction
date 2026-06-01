"""Pytest configuration and shared fixtures."""

import pytest
import torch
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_config():
    """Create a minimal config for testing."""
    from src.config import Config
    return Config({
        "data": {
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
            "split": {"train": 0.7, "val": 0.15, "test": 0.15},
            "random_seed": 42,
            "temporal_split": False
        },
        "model": {
            "input_dim": 45,
            "hidden_dims": [32, 32],
            "output_dim": 1,
            "hidden_activation": "relu",
            "output_activation": "softplus",
            "physics_penalty_weight": 0.1,
            "iv_min": 0.0,
            "iv_max": 3.0
        },
        "training": {
            "optimizer": "adam",
            "learning_rate": 0.001,
            "weight_decay": 1e-5,
            "epochs": 10,
            "batch_size": 32,
            "validation_frequency": 5,
            "use_early_stopping": False,
            "device": "cpu"
        }
    })


@pytest.fixture
def sample_data():
    """Create sample batch of data for testing."""
    batch_size = 32
    input_dim = 45
    
    X = torch.randn(batch_size, input_dim, dtype=torch.float32)
    y = torch.randn(batch_size, 1, dtype=torch.float32)
    
    return X, y


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame with IV data."""
    import pandas as pd
    
    n_samples = 100
    data = {
        "underlying": np.random.uniform(90, 110, n_samples),
        "strike_price": np.random.uniform(90, 110, n_samples),
        "option_type": np.random.choice([0, 1], n_samples),
        "iv": np.random.uniform(0.1, 0.5, n_samples),
        "expiry_encoded": np.random.randint(1, 10, n_samples)
    }
    
    # Add X0-X41 features
    for i in range(42):
        data[f"X{i}"] = np.random.randn(n_samples)
    
    return pd.DataFrame(data)
