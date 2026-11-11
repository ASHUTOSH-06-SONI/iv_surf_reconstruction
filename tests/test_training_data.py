import numpy as np
import pandas as pd
import torch

from src.config import Config
from src.data.feature_scaling import MarketFeatureScaler
from src.train.train_and_save import load_data, split_indices_by_timestamp


def test_split_indices_by_timestamp_keeps_groups_and_orders_time():
    timestamps = np.array([3, 0, 3, 0, 2, 1, 2, 1])

    train_idx, val_idx, test_idx = split_indices_by_timestamp(
        timestamps,
        train_ratio=0.5,
        val_ratio=0.25,
        random_seed=42,
        temporal=True,
    )

    assert set(timestamps[train_idx]) == {0, 1}
    assert set(timestamps[val_idx]) == {2}
    assert set(timestamps[test_idx]) == {3}
    assert set(train_idx).isdisjoint(val_idx)
    assert set(train_idx).isdisjoint(test_idx)
    assert set(val_idx).isdisjoint(test_idx)


def test_load_data_scales_from_training_and_keeps_iv_units(tmp_path):
    rows = []
    for timestamp in (3, 0, 2, 1):
        for offset in (0, 1):
            row = {
                "timestamp": timestamp,
                "underlying": 100 + timestamp,
                "option_type": "call" if offset == 0 else "put",
                "strike_price": 100 + offset,
                "iv": 5.0 if timestamp == 0 and offset == 0 else 0.2 + 0.1 * offset,
            }
            row.update({f"X{i}": timestamp * 10 + offset if i == 0 else 0.0 for i in range(42)})
            rows.append(row)

    train_file = tmp_path / "train.csv"
    pd.DataFrame(rows).to_csv(train_file, index=False)
    config = Config({
        "data": {
            "train_file": str(train_file),
            "split": {"train": 0.5, "val": 0.25, "test": 0.25},
            "random_seed": 42,
            "temporal_split": True,
        },
        "model": {"iv_min": 0.0, "iv_max": 3.0},
    })

    X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_data(
        config, torch.device("cpu")
    )

    assert X_train.shape == (3, 45)
    assert X_val.shape == (2, 45)
    assert X_test.shape == (2, 45)
    assert isinstance(scaler, MarketFeatureScaler)
    assert scaler.scaler.center_[3] == 10.0
    assert torch.allclose(
        torch.sort(y_train.flatten()).values,
        torch.tensor([0.2, 0.3, 0.3]),
    )
    assert torch.allclose(y_val.flatten(), torch.tensor([0.2, 0.3]))
    assert torch.allclose(y_test.flatten(), torch.tensor([0.2, 0.3]))
