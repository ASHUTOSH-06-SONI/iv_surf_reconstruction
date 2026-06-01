"""Configuration management for IV surface reconstruction project."""

import os
import yaml
from typing import Any, Dict, Optional
from pathlib import Path


class Config:
    """Configuration container with hierarchical attribute access."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize config from dictionary.
        
        Args:
            config_dict: Dictionary of configuration values
        """
        self._config = config_dict
        self._update_attrs(config_dict)
    
    def _update_attrs(self, config_dict: Dict[str, Any]):
        """Recursively set config values as attributes."""
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def __repr__(self) -> str:
        """String representation of config."""
        return f"Config({self._config})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config back to dictionary."""
        result = {}
        for key, value in self._config.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    def update(self, updates: Dict[str, Any]):
        """Update config with new values.
        
        Args:
            updates: Dictionary of updates to apply
        """
        def deep_update(d: Dict, u: Dict):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    deep_update(d[k], v)
                else:
                    d[k] = v
        
        deep_update(self._config, updates)
        self._update_attrs(self._config)


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config YAML file. If None, uses default.yaml
        
    Returns:
        Config object with loaded configuration
        
    Raises:
        FileNotFoundError: If config file does not exist
        yaml.YAMLError: If config file is invalid YAML
    """
    if config_path is None:
        # Use default config in config/ directory
        config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    if config_dict is None:
        config_dict = {}
    
    # Create necessary directories if they don't exist
    _create_directories(config_dict)
    
    return Config(config_dict)


def _create_directories(config_dict: Dict[str, Any]):
    """Create necessary directories specified in config.
    
    Args:
        config_dict: Configuration dictionary
    """
    dirs_to_create = []
    
    # Collect directory paths
    if "data" in config_dict and isinstance(config_dict["data"], dict):
        for key in ["raw_dir", "processed_dir"]:
            if key in config_dict["data"]:
                dirs_to_create.append(config_dict["data"][key])
    
    if "training" in config_dict and isinstance(config_dict["training"], dict):
        if "checkpoint_dir" in config_dict["training"]:
            dirs_to_create.append(config_dict["training"]["checkpoint_dir"])
    
    if "evaluation" in config_dict and isinstance(config_dict["evaluation"], dict):
        if "plot_dir" in config_dict["evaluation"]:
            dirs_to_create.append(config_dict["evaluation"]["plot_dir"])
    
    if "logging" in config_dict and isinstance(config_dict["logging"], dict):
        log_file = config_dict["logging"].get("log_file")
        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir:
                dirs_to_create.append(log_dir)
    
    # Create directories
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Simple test
    cfg = load_config()
    print(cfg)
    print(f"Model input dim: {cfg.model.input_dim}")
    print(f"Training epochs: {cfg.training.epochs}")
