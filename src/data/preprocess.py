"""Data preprocessing and cleaning utilities."""

import logging
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)


def clean_iv_column(
    df: pd.DataFrame,
    target_col: str = "iv",
    outlier_bounds: Tuple[float, float] = (-10000, 10000)
) -> pd.DataFrame:
    """Remove IV outliers outside reasonable bounds.
    
    Args:
        df: DataFrame with IV column
        target_col: Name of IV column
        outlier_bounds: (min, max) bounds for IV values
    
    Returns:
        DataFrame with outliers removed
    """
    initial_count = len(df)
    df = df[
        (df[target_col] > outlier_bounds[0]) & 
        (df[target_col] < outlier_bounds[1])
    ]
    removed = initial_count - len(df)
    if removed > 0:
        logger.info(f"Removed {removed} IV outliers")
    return df


def encode_categorical_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Encode categorical features (option_type, expiry).
    
    Args:
        df: DataFrame with categorical features
    
    Returns:
        Tuple of (encoded_df, encoders_dict)
    """
    encoders = {}
    
    # Encode option_type: call=0, put=1
    if "option_type" in df.columns:
        df = df.copy()
        df["option_type"] = df["option_type"].map({"call": 0, "put": 1})
    
    # Encode expiry as numeric
    if "expiry" in df.columns:
        df = df.copy()
        le_expiry = LabelEncoder()
        df["expiry_encoded"] = le_expiry.fit_transform(df["expiry"])
        encoders["expiry"] = le_expiry
        # Normalize expiry encoding to [0, 1]
        df["time_to_expiry"] = df["expiry_encoded"] / df["expiry_encoded"].max()
    
    return df, encoders


def preprocess_data(
    df: pd.DataFrame,
    feature_cols: Optional[list] = None,
    target_col: str = "iv",
    test_size: float = 0.2,
    random_state: int = 42,
    standardize: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, dict]:
    """Full preprocessing pipeline for IV data.
    
    Steps:
    1. Drop missing values
    2. Remove IV outliers
    3. Encode categorical features
    4. Split into train/test
    5. Standardize features
    
    Args:
        df: Input DataFrame
        feature_cols: List of feature columns. If None, uses X0-X41 + option metadata
        target_col: Name of target column (IV)
        test_size: Proportion of data for test set
        random_state: Random seed for reproducibility
        standardize: Whether to standardize features
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, scaler, metadata_dict)
    """
    if feature_cols is None:
        feature_cols = ["underlying", "option_type", "strike_price", "expiry_encoded"] + \
                      [f"X{i}" for i in range(42)]
    
    df = df.copy()
    
    # Step 1: Drop missing IV values
    initial_rows = len(df)
    df = df.dropna(subset=[target_col])
    logger.info(f"Dropped {initial_rows - len(df)} rows with missing target")
    
    # Step 2: Clean outliers
    df = clean_iv_column(df, target_col)
    
    # Step 3: Encode categorical features
    df, encoders = encode_categorical_features(df)
    
    # Step 4: Extract features and target
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )
    
    # Step 5: Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    metadata = {
        "feature_cols": feature_cols,
        "target_col": target_col,
        "encoders": encoders,
        "y_mean": y_train.mean(),
        "y_std": y_train.std(),
        "n_features": len(feature_cols),
        "n_train": len(X_train),
        "n_test": len(X_test)
    }
    
    logger.info(f"Preprocessing complete: {metadata['n_train']} train, {metadata['n_test']} test")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, metadata


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Load sample data
    df = pd.read_csv("data/processed/data.csv")
    X_train, X_test, y_train, y_test, scaler, metadata = preprocess_data(df)
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
