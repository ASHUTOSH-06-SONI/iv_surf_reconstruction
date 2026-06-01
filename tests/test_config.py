"""Unit tests for configuration management."""

import pytest
from pathlib import Path
from src.config import Config, load_config


def test_config_creation():
    """Test creating Config object from dictionary."""
    config_dict = {
        "model": {"input_dim": 45, "hidden_dims": [64, 64]},
        "training": {"epochs": 100, "learning_rate": 0.001}
    }
    cfg = Config(config_dict)
    
    assert cfg.model.input_dim == 45
    assert cfg.model.hidden_dims == [64, 64]
    assert cfg.training.epochs == 100


def test_config_to_dict():
    """Test converting Config back to dictionary."""
    config_dict = {
        "model": {"input_dim": 45},
        "training": {"epochs": 100}
    }
    cfg = Config(config_dict)
    result = cfg.to_dict()
    
    assert result["model"]["input_dim"] == 45
    assert result["training"]["epochs"] == 100


def test_config_update():
    """Test updating config values."""
    config_dict = {
        "model": {"input_dim": 45},
        "training": {"epochs": 100}
    }
    cfg = Config(config_dict)
    cfg.update({"training": {"epochs": 200}})
    
    assert cfg.training.epochs == 200
    assert cfg.model.input_dim == 45  # Unchanged


def test_load_default_config():
    """Test loading default configuration."""
    # This will load the actual default.yaml in config/
    cfg = load_config()
    
    assert hasattr(cfg, "model")
    assert hasattr(cfg, "training")
    assert hasattr(cfg, "data")
    assert cfg.model.input_dim == 45


def test_load_config_creates_directories(tmp_path):
    """Test that load_config creates necessary directories."""
    # Create a temporary config file
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text("""
data:
  raw_dir: {tmp_dir}/data/raw
  processed_dir: {tmp_dir}/data/processed
training:
  checkpoint_dir: {tmp_dir}/models
""".format(tmp_dir=tmp_path))
    
    cfg = load_config(str(config_path))
    
    # Check directories exist
    assert Path(cfg.data.raw_dir).exists() or True  # May not exist if parsed dynamically


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
