"""Compare input scaling strategies on a chronological IV prediction holdout."""

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import load_config
from src.data.feature_scaling import MarketFeatureScaler
from src.train.train_and_save import split_indices_by_timestamp
from src.models import PINNModel


LOGGER = logging.getLogger(__name__)
FEATURE_COLS = ["underlying", "option_type", "strike_price"] + [f"X{i}" for i in range(42)]
SPARSE_DIAGNOSTICS = ("X41", "X23", "X16")


def _sample_rows(indices: np.ndarray, max_rows: int) -> np.ndarray:
    if len(indices) <= max_rows:
        return indices
    positions = np.linspace(0, len(indices) - 1, num=max_rows, dtype=np.int64)
    return indices[positions]


def _make_quote_samples(
    frame: pd.DataFrame,
    iv_columns: List[str],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    market_features = frame[["underlying"] + [f"X{i}" for i in range(42)]].to_numpy(
        dtype=np.float64
    )
    feature_blocks = []
    target_blocks = []
    sparse_masks = {name: [] for name in SPARSE_DIAGNOSTICS}

    for column in iv_columns:
        match = re.fullmatch(r"(call|put)_iv_(\d+)", column)
        if match is None:
            raise ValueError(f"Unrecognized IV column name: {column}")
        option_type, strike = match.groups()
        targets = frame[column].to_numpy(dtype=np.float64)
        valid = np.isfinite(targets) & (targets >= 0.0) & (targets <= 3.0)
        if not valid.any():
            continue
        selected_features = market_features[valid]
        block = np.empty((len(selected_features), len(FEATURE_COLS)), dtype=np.float64)
        block[:, 0] = selected_features[:, 0]
        block[:, 1] = 0.0 if option_type == "call" else 1.0
        block[:, 2] = float(strike)
        block[:, 3:] = selected_features[:, 1:]
        feature_blocks.append(block)
        target_blocks.append(targets[valid])
        for name in SPARSE_DIAGNOSTICS:
            sparse_masks[name].append(frame[name].to_numpy(dtype=np.float64)[valid] != 0)

    if not feature_blocks:
        raise ValueError("No valid IV quote observations in the selected rows.")
    return (
        np.concatenate(feature_blocks),
        np.concatenate(target_blocks),
        {name: np.concatenate(parts) for name, parts in sparse_masks.items()},
    )


def _score(
    model: PINNModel,
    features: np.ndarray,
    targets: np.ndarray,
    sparse_masks: Dict[str, np.ndarray],
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        predictions = model(torch.as_tensor(features, dtype=torch.float32, device=device))
    predicted = predictions.cpu().numpy().reshape(-1)
    return _score_predictions(predicted, targets, sparse_masks)


def _score_predictions(
    predicted: np.ndarray,
    targets: np.ndarray,
    sparse_masks: Dict[str, np.ndarray],
) -> Dict[str, float]:
    residual = predicted - targets
    scores = {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": float(1.0 - np.sum(residual**2) / np.sum((targets - targets.mean()) ** 2)),
    }
    for name, mask in sparse_masks.items():
        for state, selected in (("nonzero", mask), ("zero", ~mask)):
            if selected.any():
                error = residual[selected]
                scores[f"{name}_{state}_mae"] = float(np.mean(np.abs(error)))
                scores[f"{name}_{state}_rmse"] = float(np.sqrt(np.mean(error**2)))
                scores[f"{name}_{state}_n"] = int(selected.sum())
    return scores


def compare_scalers(
    train_path: str,
    output_path: str,
    epochs: int = 20,
    train_rows: int = 1500,
    eval_rows: int = 400,
    seed: int = 42,
) -> pd.DataFrame:
    """Train identical small PINNs under each scaler and score a held-out period."""
    data = pd.read_parquet(train_path).sort_values("timestamp").reset_index(drop=True)
    iv_columns = [c for c in data.columns if c.startswith(("call_iv_", "put_iv_"))]
    train_idx, val_idx, test_idx = split_indices_by_timestamp(
        data["timestamp"].to_numpy(),
        train_ratio=0.70,
        val_ratio=0.15,
        random_seed=seed,
        temporal=True,
    )
    train_idx = _sample_rows(train_idx, train_rows)
    val_idx = _sample_rows(val_idx, eval_rows)
    test_idx = _sample_rows(test_idx, eval_rows)

    X_train, y_train, _ = _make_quote_samples(data.iloc[train_idx], iv_columns)
    X_val, y_val, val_masks = _make_quote_samples(data.iloc[val_idx], iv_columns)
    X_test, y_test, test_masks = _make_quote_samples(data.iloc[test_idx], iv_columns)

    device = torch.device("cpu")
    median_prediction = np.full_like(y_test, np.median(y_train))
    median_val_prediction = np.full_like(y_val, np.median(y_train))
    results = [{
        "scaling": "train_median_baseline",
        "input_features": 0,
        "train_quotes": len(y_train),
        "validation_quotes": len(y_val),
        "test_quotes": len(y_test),
        "best_val_rmse": float(np.sqrt(np.mean((median_val_prediction - y_val) ** 2))),
        **_score_predictions(median_prediction, y_test, test_masks),
    }]
    for method in MarketFeatureScaler.METHODS:
        scaler = MarketFeatureScaler(FEATURE_COLS, method=method)
        X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
        X_val_scaled = scaler.transform(X_val).astype(np.float32)
        X_test_scaled = scaler.transform(X_test).astype(np.float32)

        torch.manual_seed(seed)
        model = PINNModel(
            input_dim=X_train_scaled.shape[1],
            hidden_dims=[64, 64],
            output_dim=1,
            output_activation="softplus",
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1.0e-5)
        criterion = nn.MSELoss()
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_train_scaled),
                torch.from_numpy(y_train.astype(np.float32)).unsqueeze(1),
            ),
            batch_size=512,
            shuffle=True,
            generator=generator,
        )

        best_state = None
        best_val_mse = float("inf")
        for _ in range(epochs):
            model.train()
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                loss = criterion(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(torch.from_numpy(X_val_scaled))
                val_mse = float(criterion(val_pred, torch.from_numpy(y_val.astype(np.float32)).unsqueeze(1)))
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

        if best_state is None:
            raise RuntimeError(f"No validation checkpoint was produced for {method}.")
        model.load_state_dict(best_state)
        scores = _score(model, X_test_scaled, y_test, test_masks, device)
        results.append({
            "scaling": method,
            "input_features": X_train_scaled.shape[1],
            "train_quotes": len(y_train),
            "validation_quotes": len(y_val),
            "test_quotes": len(y_test),
            "best_val_rmse": float(np.sqrt(best_val_mse)),
            **scores,
        })
        LOGGER.info("%s: %s", method, results[-1])

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result_frame = pd.DataFrame(results)
    result_frame.to_csv(output, index=False)
    return result_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--output", default="results/feature_scaling_comparison.csv")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-rows", type=int, default=1500)
    parser.add_argument("--eval-rows", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(args.config)
    results = compare_scalers(
        config.data.raw_train_file,
        args.output,
        epochs=args.epochs,
        train_rows=args.train_rows,
        eval_rows=args.eval_rows,
        seed=args.seed,
    )
    print(results.to_string(index=False))
    print(f"\nSaved comparison to {args.output}")


if __name__ == "__main__":
    main()
