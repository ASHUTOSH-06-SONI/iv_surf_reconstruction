"""Unit tests for data preprocessing."""

import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from src.data.preprocess import clean_iv_column, encode_categorical_features, preprocess_data


def test_clean_iv_column(sample_dataframe):
    """Test IV column cleaning."""
    # Add some outliers
    df = sample_dataframe.copy()
    df.loc[0, "iv"] = -5000
    df.loc[1, "iv"] = 50000
    
    cleaned = clean_iv_column(df, target_col="iv", outlier_bounds=(0, 3))
    
    assert len(cleaned) == len(df) - 2
    assert cleaned["iv"].min() >= 0
    assert cleaned["iv"].max() <= 3


def test_clean_iv_column_no_outliers(sample_dataframe):
    """Test cleaning when no outliers exist."""
    original_len = len(sample_dataframe)
    cleaned = clean_iv_column(sample_dataframe)
    
    assert len(cleaned) == original_len


def test_encode_categorical_features():
    """Test categorical feature encoding."""
    df = pd.DataFrame({
        "option_type": ["call", "put", "call"],
        "expiry": ["2024-06", "2024-06", "2024-09"]
    })
    
    encoded_df, encoders = encode_categorical_features(df)
    
    # Check option_type encoding
    assert all(encoded_df["option_type"].isin([0, 1]))
    
    # Check expiry encoding
    assert "expiry_encoded" in encoded_df.columns
    assert "time_to_expiry" in encoded_df.columns


def test_preprocess_data(sample_dataframe):
    """Test full preprocessing pipeline."""
    feature_cols = ["underlying", "option_type", "strike_price"] + [f"X{i}" for i in range(42)]
    
    X_train, X_test, y_train, y_test, scaler, metadata = preprocess_data(
        sample_dataframe,
        feature_cols=feature_cols,
        test_size=0.2,
        random_state=42
    )
    
    # Check shapes
    assert X_train.shape[1] == len(feature_cols)
    assert X_test.shape[1] == len(feature_cols)
    assert len(y_train) + len(y_test) == len(sample_dataframe)
    
    # Check metadata
    assert metadata["n_train"] + metadata["n_test"] == len(sample_dataframe)
    assert metadata["n_features"] == len(feature_cols)


def test_preprocess_data_removes_missing_values():
    """Test that preprocessing removes rows with missing target."""
    df = pd.DataFrame({
        "underlying": [100, 100, 100],
        "option_type": [0, 0, 0],
        "strike_price": [100, 100, 100],
        "iv": [0.2, np.nan, 0.3],
    })
    
    # Add X features
    for i in range(42):
        df[f"X{i}"] = [1, 1, 1]
    
    X_train, X_test, y_train, y_test, scaler, metadata = preprocess_data(df)
    
    # Should only have 2 samples (one removed due to missing IV)
    assert metadata["n_train"] + metadata["n_test"] == 2


def test_preprocess_robust_scaling(sample_dataframe):
    """Test that features are robust-scaled using training data."""
    X_train, X_test, _, _, scaler, _ = preprocess_data(sample_dataframe)
    
    assert isinstance(scaler, RobustScaler)
    train_median = np.median(X_train, axis=0)
    train_iqr = np.quantile(X_train, 0.75, axis=0) - np.quantile(X_train, 0.25, axis=0)
    
    np.testing.assert_allclose(train_median, np.zeros(X_train.shape[1]), atol=0.15)
    np.testing.assert_allclose(train_iqr, np.ones(X_train.shape[1]), atol=0.15)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
