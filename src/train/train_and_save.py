"""Training script for PINN model with physics constraints."""

import logging
import os
from pathlib import Path
from typing import Tuple, Optional
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.config import Config
from src.data.feature_scaling import MarketFeatureScaler
from src.data.preprocess import clean_iv_column, prepare_processed_data
from src.models import PINNModel, physics_penalty


logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping callback to prevent overfitting.
    
    Stops training if validation metric doesn't improve for a specified patience.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs without improvement before stopping
            min_delta: Minimum change in metric to count as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_metric = None
        self.best_epoch = 0
    
    def __call__(self, metric: float, epoch: int) -> bool:
        """Check if training should stop.
        
        Args:
            metric: Current validation metric value
            epoch: Current epoch number
        
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_metric is None:
            self.best_metric = metric
            self.best_epoch = epoch
        elif metric < self.best_metric - self.min_delta:
            self.best_metric = metric
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logger.info(
                    f"Early stopping triggered. Best metric: {self.best_metric:.6f} "
                    f"at epoch {self.best_epoch}"
                )
                return True
        return False


def get_device(device_name: str) -> torch.device:
    """Resolve compute device name to a torch.device.

    Args:
        device_name: One of "cpu", "cuda", "mps", or "auto".

    Returns:
        Resolved torch.device.
    """
    if device_name == "auto":
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if device_name == "mps":
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("Requested MPS device, but MPS is not available on this system.")

    if device_name == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("Requested CUDA device, but CUDA is not available on this system.")

    return torch.device(device_name)


def split_indices_by_timestamp(
    timestamps: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    random_seed: int,
    temporal: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split rows by unique timestamp so quotes from one surface stay together."""
    if pd.isna(timestamps).any():
        raise ValueError("Timestamps must not be missing when splitting the dataset.")
    if len(timestamps) == 0:
        raise ValueError("Cannot split a dataset with no timestamps.")
    _, group_indices = np.unique(timestamps, return_inverse=True)
    n_groups = int(group_indices.max()) + 1
    n_train = int(n_groups * train_ratio)
    n_val = int(n_groups * val_ratio)
    if n_train == 0 or n_val == 0 or n_groups - n_train - n_val == 0:
        raise ValueError("Train, validation, and test splits must each contain timestamps.")

    group_order = np.arange(n_groups)
    if not temporal:
        np.random.default_rng(random_seed).shuffle(group_order)

    train_groups = group_order[:n_train]
    val_groups = group_order[n_train:n_train + n_val]
    test_groups = group_order[n_train + n_val:]
    group_splits = np.empty(n_groups, dtype=np.int8)
    group_splits[train_groups] = 0
    group_splits[val_groups] = 1
    group_splits[test_groups] = 2
    row_splits = group_splits[group_indices]
    return (
        np.flatnonzero(row_splits == 0),
        np.flatnonzero(row_splits == 1),
        np.flatnonzero(row_splits == 2),
    )


def load_data(
    config: Config,
    device: torch.device
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    MarketFeatureScaler,
]:
    """Load and preprocess data from CSV files.
    
    Args:
        config: Configuration object with data paths and split ratios
        device: Torch device on which tensors should be allocated
    
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test, scaler)
    """
    logger.info(f"Loading training data from {config.data.train_file}")
    
    # Prepare processed dataset from raw parquet if the processed file is missing.
    if not Path(config.data.train_file).exists():
        raw_path = getattr(config.data, "raw_train_file", None)
        if raw_path and Path(raw_path).exists():
            logger.info(f"Processed training file missing; preparing from raw parquet: {raw_path}")
            prepare_processed_data(
                raw_train_file=raw_path,
                output_train_file=config.data.train_file
            )
        else:
            raise FileNotFoundError(f"Training data not found: {config.data.train_file}")
    
    train_df = pd.read_csv(config.data.train_file)
    
    # Drop missing IV values and exclude only quotes outside configured physical bounds.
    initial_rows = len(train_df)
    train_df = train_df.dropna(subset=["iv"])
    logger.info(f"Dropped {initial_rows - len(train_df)} rows with missing IV values")

    train_df = clean_iv_column(
        train_df,
        target_col="iv",
        outlier_bounds=(config.model.iv_min, config.model.iv_max),
    )
    logger.info(f"Training set shape: {train_df.shape}")
    
    # Extract features and target
    feature_cols = ["underlying", "option_type", "strike_price"] + [f"X{i}" for i in range(42)]
    if "timestamp" not in train_df:
        raise ValueError("Processed training data must include timestamps for leakage-safe splitting.")
    train_df = train_df.copy()
    train_df["option_type"] = train_df["option_type"].map({"call": 0.0, "put": 1.0})
    if train_df["option_type"].isna().any():
        raise ValueError("Processed data contains an unknown option_type; expected 'call' or 'put'.")
    X = train_df[feature_cols].to_numpy(dtype=np.float64)
    if not np.isfinite(X).all():
        raise ValueError("Processed feature values must all be finite.")
    y = train_df["iv"].values
    
    train_idx, val_idx, test_idx = split_indices_by_timestamp(
        train_df["timestamp"].to_numpy(),
        train_ratio=config.data.split.train,
        val_ratio=config.data.split.val,
        random_seed=config.data.random_seed,
        temporal=config.data.temporal_split,
    )

    # Fit feature transformations on training rows only. IV targets remain in decimal units.
    scaling_method = getattr(config.data, "feature_scaling", "robust")
    scaler = MarketFeatureScaler(feature_cols, method=scaling_method)
    X_train = scaler.fit_transform(X[train_idx])
    X_val = scaler.transform(X[val_idx])
    X_test = scaler.transform(X[test_idx])
    
    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y[train_idx], dtype=torch.float32).to(device).unsqueeze(1)
    
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y[val_idx], dtype=torch.float32).to(device).unsqueeze(1)
    
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y[test_idx], dtype=torch.float32).to(device).unsqueeze(1)
    
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


def train_model(config: Config, save_best_model: bool = True) -> dict:
    """Train PINN model with physics constraints.
    
    Args:
        config: Configuration object with model and training parameters
        save_best_model: Whether to save the best model checkpoint
    
    Returns:
        Dictionary containing training history and metadata
    """
    logger.info("=" * 80)
    logger.info("Starting PINN Training")
    logger.info("=" * 80)
    
    # Setup device
    device = get_device(config.training.device)
    logger.info(f"Using device: {device}")
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_data(config, device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True
    )
    
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False)
    
    # Initialize model
    model = PINNModel(
        input_dim=X_train.shape[1],
        hidden_dims=config.model.hidden_dims,
        output_dim=config.model.output_dim,
        hidden_activation=config.model.hidden_activation,
        output_activation=config.model.output_activation
    ).to(device)
    
    logger.info(f"Model:\n{model}")
    
    # Optimizer and loss function
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    criterion = nn.MSELoss()
    
    # Early stopping
    early_stopping = None
    if config.training.use_early_stopping:
        early_stopping = EarlyStopping(patience=config.training.early_stopping_patience)
    
    # Training loop
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_mse": [],
        "val_mse": [],
        "epoch": []
    }
    
    logger.info("=" * 80)
    logger.info("Training Progress")
    logger.info("=" * 80)
    
    for epoch in range(config.training.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_mse = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            pred_iv = model(batch_X)
            
            # Loss computation
            loss_supervised = criterion(pred_iv, batch_y)
            loss_physics = physics_penalty(
                pred_iv,
                iv_min=config.model.iv_min,
                iv_max=config.model.iv_max
            )
            loss = loss_supervised + config.model.physics_penalty_weight * loss_physics
            
            # Backprop
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_mse += loss_supervised.item()
        
        train_loss /= len(train_loader)
        train_mse /= len(train_loader)
        
        # Validation phase
        if (epoch + 1) % config.training.validation_frequency == 0:
            model.eval()
            val_loss = 0.0
            val_mse = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    pred_iv = model(batch_X)
                    
                    loss_supervised = criterion(pred_iv, batch_y)
                    loss_physics = physics_penalty(
                        pred_iv,
                        iv_min=config.model.iv_min,
                        iv_max=config.model.iv_max
                    )
                    loss = loss_supervised + config.model.physics_penalty_weight * loss_physics
                    
                    val_loss += loss.item()
                    val_mse += loss_supervised.item()
            
            val_loss /= len(val_loader)
            val_mse /= len(val_loader)
            
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["train_mse"].append(train_mse)
            history["val_loss"].append(val_loss)
            history["val_mse"].append(val_mse)
            
            # Logging
            if (epoch + 1) % (config.training.validation_frequency * 5) == 0:
                logger.info(
                    f"Epoch {epoch:4d} | "
                    f"Train Loss: {train_loss:.6f} | Train MSE: {train_mse:.6f} | "
                    f"Val Loss: {val_loss:.6f} | Val MSE: {val_mse:.6f}"
                )
            
            # Early stopping
            if early_stopping and early_stopping(val_loss, epoch):
                break
            
            # Save best model
            if save_best_model and (epoch == config.training.validation_frequency - 1 or 
                                   val_loss == min(history["val_loss"])):
                checkpoint_path = Path(config.training.best_model_path)
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": config.to_dict()
                }, checkpoint_path)
    
    logger.info("=" * 80)
    logger.info(f"Training completed. Best model saved to {config.training.best_model_path}")
    logger.info("=" * 80)
    
    # Save feature scaling metadata; targets and model outputs use IV decimal units.
    metadata_path = Path(config.training.best_model_path).parent / "metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump({
            "scaler": scaler,
            "input_dim": X_train.shape[1],
            "feature_cols": scaler.feature_names_out,
            "feature_scaling": scaler.method,
            "target_units": "implied_volatility_decimal",
        }, f)
    logger.info(f"Metadata saved to {metadata_path}")
    
    return history


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load config and train
    from src.config import load_config
    config = load_config()
    history = train_model(config)
