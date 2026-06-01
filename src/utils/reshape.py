"""Data reshaping utilities for IV surface reconstruction.

Converts option data from wide format (columns for each strike) to long format
(one row per strike observation).
"""

import pandas as pd
from pathlib import Path
from typing import Tuple


def reshape_wide_to_long(
    df: pd.DataFrame,
    feature_cols: list = None,
    call_iv_pattern: str = "call_iv_",
    put_iv_pattern: str = "put_iv_",
    strike_pattern: str = r'(\d+)'
) -> pd.DataFrame:
    """Convert wide format option data to long format.
    
    Wide format: Each strike has its own column (call_iv_500, call_iv_510, etc.)
    Long format: One row per (underlying, strike, option_type) with IV as value
    
    Args:
        df: DataFrame with wide format option data
        feature_cols: Columns to keep as ID columns (e.g., ["timestamp", "underlying", "expiry"])
        call_iv_pattern: Pattern to identify call IV columns
        put_iv_pattern: Pattern to identify put IV columns
        strike_pattern: Regex pattern to extract numeric strike from column name
    
    Returns:
        DataFrame in long format with columns:
            [feature_cols, "option_type", "strike_price", "iv"]
    
    Examples:
        >>> df_long = reshape_wide_to_long(df_wide, feature_cols=["timestamp", "underlying"])
    """
    if feature_cols is None:
        feature_cols = ["timestamp", "underlying", "expiry"] + [f"X{i}" for i in range(42)]
    
    # Identify IV columns
    call_iv_cols = [col for col in df.columns if col.startswith(call_iv_pattern)]
    put_iv_cols = [col for col in df.columns if col.startswith(put_iv_pattern)]
    
    if not call_iv_cols and not put_iv_cols:
        raise ValueError(
            f"No IV columns found with patterns '{call_iv_pattern}' or '{put_iv_pattern}'"
        )
    
    # Melt call options
    call_df = df.melt(
        id_vars=feature_cols,
        value_vars=call_iv_cols,
        var_name="strike",
        value_name="iv"
    )
    call_df["option_type"] = "call"
    
    # Melt put options
    put_df = df.melt(
        id_vars=feature_cols,
        value_vars=put_iv_cols,
        var_name="strike",
        value_name="iv"
    )
    put_df["option_type"] = "put"
    
    # Combine
    long_df = pd.concat([call_df, put_df], ignore_index=True)
    
    # Extract numeric strike from column name
    long_df["strike_price"] = long_df["strike"].str.extract(strike_pattern).astype(int)
    long_df.drop(columns=["strike"], inplace=True)
    
    return long_df


def reshape_from_file(
    input_path: str,
    output_path: str,
    feature_cols: list = None
) -> pd.DataFrame:
    """Load data from file, reshape, and save to new file.
    
    Args:
        input_path: Path to input parquet or CSV file
        output_path: Path to save reshaped CSV file
        feature_cols: Columns to keep as ID columns
    
    Returns:
        Reshaped DataFrame
    """
    # Load data
    if str(input_path).endswith(".parquet"):
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)
    
    # Reshape
    long_df = reshape_wide_to_long(df, feature_cols=feature_cols)
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(output_path, index=False)
    
    return long_df


if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "reshaped_data.csv"
        reshape_from_file(input_file, output_file)
        print(f"✅ Reshaped data saved to {output_file}")
