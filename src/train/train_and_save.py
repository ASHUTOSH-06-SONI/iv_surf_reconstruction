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
from sklearn.preprocessing import StandardScaler

from src.config import Config
from src.data.preprocess import prepare_processed_data
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


def load_data(config: Config) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, 
                                       torch.Tensor, torch.Tensor, StandardScaler]:
    """Load and preprocess data from CSV files.
    
    Args:
        config: Configuration object with data paths and split ratios
    
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
    
    # Drop missing IV values
    initial_rows = len(train_df)
    train_df = train_df.dropna(subset=["iv"])
    logger.info(f"Dropped {initial_rows - len(train_df)} rows with missing IV values")
    
    # Filter IV outliers
    train_df = train_df[
        (train_df["iv"] >= config.model.iv_min) & 
        (train_df["iv"] <= config.model.iv_max)
    ]
    logger.info(f"Training set shape: {train_df.shape}")
    
    # Extract features and target
    feature_cols = ["underlying", "option_type", "strike_price"] + [f"X{i}" for i in range(42)]
    X = train_df[feature_cols].values
    y = train_df["iv"].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Normalize target (normalize IV to mean=0, std=1)
    y_mean = y.mean()
    y_std = y.std()
    y_norm = (y - y_mean) / y_std
    
    # Create train/val/test split
    np.random.seed(config.data.random_seed)
    n_samples = len(X_scaled)
    
    indices = np.arange(n_samples)
    if config.data.temporal_split:
        # Use first 70% for train, next 15% for val, last 15% for test
        train_idx = indices[:int(n_samples * config.data.split.train)]
        val_idx = indices[
            int(n_samples * config.data.split.train):
            int(n_samples * (config.data.split.train + config.data.split.val))
        ]
        test_idx = indices[int(n_samples * (config.data.split.train + config.data.split.val)):]
    else:
        # Random split
        np.random.shuffle(indices)
        train_idx = indices[:int(n_samples * config.data.split.train)]
        val_idx = indices[
            int(n_samples * config.data.split.train):
            int(n_samples * (config.data.split.train + config.data.split.val))
        ]
        test_idx = indices[int(n_samples * (config.data.split.train + config.data.split.val)):]
    
    # Convert to tensors
    device = torch.device(config.training.device)
    X_train = torch.tensor(X_scaled[train_idx], dtype=torch.float32).to(device)
    y_train = torch.tensor(y_norm[train_idx], dtype=torch.float32).to(device).unsqueeze(1)
    
    X_val = torch.tensor(X_scaled[val_idx], dtype=torch.float32).to(device)
    y_val = torch.tensor(y_norm[val_idx], dtype=torch.float32).to(device).unsqueeze(1)
    
    X_test = torch.tensor(X_scaled[test_idx], dtype=torch.float32).to(device)
    y_test = torch.tensor(y_norm[test_idx], dtype=torch.float32).to(device).unsqueeze(1)
    
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, y_mean, y_std


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
    device = torch.device(config.training.device)
    logger.info(f"Using device: {device}")
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, y_mean, y_std = load_data(config)
    
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
        input_dim=config.model.input_dim,
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
            pred_normalized = model(batch_X)
            
            # Unnormalized predictions for physics penalty
            pred_unnormalized = pred_normalized * y_std + y_mean
            
            # Loss computation
            loss_supervised = criterion(pred_normalized, batch_y)
            loss_physics = physics_penalty(
                pred_unnormalized,
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
                    pred_normalized = model(batch_X)
                    pred_unnormalized = pred_normalized * y_std + y_mean
                    
                    loss_supervised = criterion(pred_normalized, batch_y)
                    loss_physics = physics_penalty(
                        pred_unnormalized,
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
    
    # Save scaler and normalization parameters
    metadata_path = Path(config.training.best_model_path).parent / "metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump({
            "scaler": scaler,
            "y_mean": y_mean,
            "y_std": y_std,
            "input_dim": config.model.input_dim
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
