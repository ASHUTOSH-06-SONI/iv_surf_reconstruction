"""Benchmark IV quote reconstruction from partially observed historical surfaces."""

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from torch.utils.data import DataLoader, TensorDataset

from src.models import PINNModel, physics_penalty
from src.train.train_and_save import split_indices_by_timestamp


LOGGER = logging.getLogger(__name__)
MODEL_NAMES = ("surface_median", "linear_interpolation", "sabr", "surface_mlp", "pinn")


def _iv_columns(columns: Sequence[str]) -> Tuple[List[str], np.ndarray, np.ndarray]:
    iv_cols = []
    strikes = []
    is_call = []
    for column in columns:
        match = re.fullmatch(r"(call|put)_iv_(\d+)", column)
        if match:
            iv_cols.append(column)
            is_call.append(match.group(1) == "call")
            strikes.append(float(match.group(2)))
    if not iv_cols:
        raise ValueError("No call_iv_<strike> or put_iv_<strike> columns found.")
    return iv_cols, np.asarray(strikes), np.asarray(is_call, dtype=bool)


def _years_to_expiry(timestamp_ns: int, expiry: object) -> float:
    timestamp = pd.to_datetime(timestamp_ns, unit="ns", utc=True)
    expiry_timestamp = pd.Timestamp(expiry, tz="UTC")
    seconds = max((expiry_timestamp - timestamp).total_seconds(), 60.0)
    return seconds / (365.25 * 24 * 60 * 60)


def _hagan_sabr(
    forward: float,
    strike: np.ndarray,
    maturity: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> np.ndarray:
    strike = np.asarray(strike, dtype=np.float64)
    if forward <= 0 or np.any(strike <= 0):
        return np.full_like(strike, np.nan)

    log_fk = np.log(forward / strike)
    one_minus_beta = 1.0 - beta
    fk_power = (forward * strike) ** (0.5 * one_minus_beta)
    z = (nu / alpha) * fk_power * log_fk
    sqrt_term = np.sqrt(np.maximum(1.0 - 2.0 * rho * z + z * z, 0.0))
    x_z = np.log((sqrt_term + z - rho) / (1.0 - rho))
    z_over_x = np.ones_like(z)
    nonzero = np.abs(x_z) > 1e-10
    z_over_x[nonzero] = z[nonzero] / x_z[nonzero]

    log_correction = (
        (one_minus_beta**2 / 24.0) * log_fk**2
        + (one_minus_beta**4 / 1920.0) * log_fk**4
    )
    time_correction = (
        (one_minus_beta**2 / 24.0) * alpha**2 / fk_power**2
        + (rho * beta * nu * alpha / 4.0) / fk_power
        + ((2.0 - 3.0 * rho**2) / 24.0) * nu**2
    )
    return (
        alpha
        / fk_power
        * z_over_x
        * (1.0 + log_correction)
        * (1.0 + time_correction * maturity)
    )


def _fit_sabr(
    forward: float,
    strikes: np.ndarray,
    ivs: np.ndarray,
    maturity: float,
    beta: float = 0.7,
) -> np.ndarray:
    """Fit alpha/rho/nu to observed call and put quotes with beta fixed."""
    scale = forward ** (1.0 - beta)
    initial_alpha = max(float(np.median(ivs)) * scale, 1e-4)

    def residual(parameters: np.ndarray) -> np.ndarray:
        alpha, rho, nu = parameters
        predicted = _hagan_sabr(forward, strikes, maturity, alpha, beta, rho, nu)
        return np.nan_to_num(predicted - ivs, nan=1e3, posinf=1e3, neginf=-1e3)

    result = least_squares(
        residual,
        x0=np.array([initial_alpha, 0.0, 0.5]),
        bounds=(
            np.array([1e-5, -0.995, 1e-4]),
            np.array([max(scale * 5.0, 1.0), 0.995, 5.0]),
        ),
        loss="soft_l1",
        f_scale=0.05,
        max_nfev=100,
    )
    if not result.success and not np.isfinite(result.cost):
        raise RuntimeError(f"SABR calibration failed: {result.message}")
    return result.x


def _interpolate_surface(
    values: np.ndarray,
    observed: np.ndarray,
    strikes: np.ndarray,
    is_call: np.ndarray,
) -> np.ndarray:
    prediction = np.full(len(values), np.nan, dtype=np.float64)
    for call_side in (True, False):
        side = is_call == call_side
        known = side & observed & np.isfinite(values)
        target = side & ~observed
        if not target.any() or not known.any():
            continue
        x = strikes[known]
        y = values[known]
        order = np.argsort(x)
        prediction[target] = np.interp(strikes[target], x[order], y[order])
    return prediction


def _surface_features(
    frame: pd.DataFrame,
    strikes: np.ndarray,
    iv_cols: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    values = frame[list(iv_cols)].to_numpy(dtype=np.float64)
    values[(values < 0.0) | (values > 3.0)] = np.nan
    forwards = frame["underlying"].to_numpy(dtype=np.float64)
    if np.any(forwards <= 0):
        raise ValueError("Underlying values must be positive to calculate log-moneyness.")
    log_moneyness = np.log(strikes[None, :] / forwards[:, None])
    expiry_days = np.array([
        _years_to_expiry(timestamp, expiry) * 365.25
        for timestamp, expiry in zip(frame["timestamp"], frame["expiry"])
    ])
    model_inputs = np.column_stack([log_moneyness, np.log1p(expiry_days)])
    return values, model_inputs


def _sample_masks(
    values: np.ndarray,
    call_side: np.ndarray,
    test_side_counts: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    observed = np.zeros(values.shape, dtype=bool)
    hidden = np.zeros(values.shape, dtype=bool)
    for row in range(len(values)):
        valid = np.isfinite(values[row])
        for side in (True, False):
            indices = np.flatnonzero(valid & (call_side == side))
            count_index = 0 if side else 1
            n_observed = min(int(test_side_counts[row, count_index]), len(indices))
            if n_observed == 0 and len(indices):
                n_observed = 1
            selected = rng.choice(indices, size=n_observed, replace=False) if n_observed else []
            observed[row, selected] = True
        hidden[row] = valid & ~observed[row]
    return observed, hidden


class SurfaceMLP(nn.Module):
    def __init__(self, input_dim: int, n_quotes: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_quotes),
            nn.Softplus(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


def _mlp_inputs(values: np.ndarray, observed: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    masked_values = np.where(observed, values, 0.0)
    return np.concatenate([masked_values, observed.astype(np.float64), coordinates], axis=1)


def _fit_surface_mlp(
    train_values: np.ndarray,
    train_coordinates: np.ndarray,
    train_masks: np.ndarray,
    epochs: int,
    seed: int,
) -> SurfaceMLP:
    torch.manual_seed(seed)
    inputs = _mlp_inputs(train_values, train_masks, train_coordinates).astype(np.float32)
    targets = np.nan_to_num(train_values, nan=0.0).astype(np.float32)
    hidden = (~train_masks) & np.isfinite(train_values)
    dataset = TensorDataset(
        torch.from_numpy(inputs),
        torch.from_numpy(targets),
        torch.from_numpy(hidden.astype(np.float32)),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, generator=generator)
    model = SurfaceMLP(inputs.shape[1], train_values.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    for _ in range(epochs):
        model.train()
        for batch_inputs, batch_targets, batch_hidden in loader:
            optimizer.zero_grad()
            predictions = model(batch_inputs)
            squared_error = (predictions - batch_targets) ** 2 * batch_hidden
            loss = squared_error.sum() / batch_hidden.sum().clamp_min(1.0)
            loss.backward()
            optimizer.step()
    return model


def _fit_pinn(
    train_values: np.ndarray,
    train_coordinates: np.ndarray,
    train_masks: np.ndarray,
    epochs: int,
    seed: int,
) -> PINNModel:
    """Train the repository PINN on masked surfaces with an IV-range penalty."""
    torch.manual_seed(seed)
    inputs = _mlp_inputs(train_values, train_masks, train_coordinates).astype(np.float32)
    targets = np.nan_to_num(train_values, nan=0.0).astype(np.float32)
    hidden = (~train_masks) & np.isfinite(train_values)
    dataset = TensorDataset(
        torch.from_numpy(inputs),
        torch.from_numpy(targets),
        torch.from_numpy(hidden.astype(np.float32)),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, generator=generator)
    model = PINNModel(
        input_dim=inputs.shape[1],
        hidden_dims=[64, 64],
        output_dim=train_values.shape[1],
        output_activation="softplus",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1.0e-5)
    criterion = nn.MSELoss()
    for _ in range(epochs):
        model.train()
        for batch_inputs, batch_targets, batch_hidden in loader:
            optimizer.zero_grad()
            predictions = model(batch_inputs)
            supervised = ((predictions - batch_targets) ** 2 * batch_hidden).sum()
            supervised = supervised / batch_hidden.sum().clamp_min(1.0)
            constrained = physics_penalty(predictions, iv_min=0.0, iv_max=3.0)
            loss = supervised + 0.1 * constrained
            loss.backward()
            optimizer.step()
    return model


def _score(
    truth: np.ndarray,
    predictions: np.ndarray,
    hidden: np.ndarray,
) -> Dict[str, float]:
    valid = hidden & np.isfinite(truth) & np.isfinite(predictions)
    n_hidden = int((hidden & np.isfinite(truth)).sum())
    if not valid.any():
        return {
            "n_hidden": n_hidden,
            "n_scored": 0,
            "coverage": 0.0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "r2": float("nan"),
        }
    residual = predictions[valid] - truth[valid]
    total = np.sum((truth[valid] - truth[valid].mean()) ** 2)
    return {
        "n_hidden": n_hidden,
        "n_scored": int(valid.sum()),
        "coverage": float(valid.sum() / n_hidden) if n_hidden else 0.0,
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": float(1.0 - np.sum(residual**2) / total) if total > 0 else float("nan"),
    }


def _evaluate_classical(
    method: str,
    frame: pd.DataFrame,
    values: np.ndarray,
    observed: np.ndarray,
    strikes: np.ndarray,
    call_side: np.ndarray,
    hidden: np.ndarray,
) -> Tuple[np.ndarray, int]:
    predictions = np.full(values.shape, np.nan, dtype=np.float64)
    failures = 0
    for row in range(len(values)):
        known = observed[row] & np.isfinite(values[row])
        target = hidden[row]
        if not target.any():
            continue
        if method == "surface_median":
            if known.any():
                predictions[row, target] = np.median(values[row, known])
        elif method == "linear_interpolation":
            predictions[row] = _interpolate_surface(
                values[row], observed[row], strikes, call_side
            )
        elif method == "sabr":
            if known.sum() < 4:
                failures += 1
                continue
            try:
                maturity = _years_to_expiry(
                    int(frame.iloc[row]["timestamp"]), frame.iloc[row]["expiry"]
                )
                alpha, rho, nu = _fit_sabr(
                    float(frame.iloc[row]["underlying"]),
                    strikes[known],
                    values[row, known],
                    maturity,
                )
                predicted = _hagan_sabr(
                    float(frame.iloc[row]["underlying"]),
                    strikes[target],
                    maturity,
                    alpha,
                    0.7,
                    rho,
                    nu,
                )
                predictions[row, target] = np.clip(predicted, 0.0, 3.0)
            except (ValueError, FloatingPointError, RuntimeError, OverflowError):
                failures += 1
    return predictions, failures


def _plot_holdout_reconstruction(
    frame: pd.DataFrame,
    truth: np.ndarray,
    observed: np.ndarray,
    strikes: np.ndarray,
    call_side: np.ndarray,
    predictions: Dict[str, np.ndarray],
    output_path: Path,
) -> None:
    row = 0
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    colors = {
        "surface_median": "tab:gray",
        "linear_interpolation": "tab:blue",
        "sabr": "tab:orange",
        "surface_mlp": "tab:green",
        "pinn": "tab:purple",
    }
    for axis, is_call, label in ((axes[0], True, "Call IV"), (axes[1], False, "Put IV")):
        side = call_side == is_call
        axis.scatter(
            strikes[side & observed[row]],
            truth[row, side & observed[row]],
            color="black",
            marker="o",
            label="Observed input",
            zorder=4,
        )
        axis.scatter(
            strikes[side & ~observed[row] & np.isfinite(truth[row])],
            truth[row, side & ~observed[row] & np.isfinite(truth[row])],
            color="red",
            marker="x",
            label="Hidden ground truth",
            zorder=4,
        )
        for method, method_predictions in predictions.items():
            axis.plot(
                strikes[side],
                method_predictions[row, side],
                marker=".",
                linewidth=1.2,
                color=colors[method],
                label="PINN" if method == "pinn" else method.replace("_", " "),
            )
        axis.set_title(label)
        axis.set_xlabel("Strike")
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("Implied volatility")
    axes[1].legend(fontsize=8, loc="best")
    figure.suptitle(
        f"Masked-quote reconstruction example at timestamp "
        f"{frame.iloc[row]['timestamp']}"
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def run_benchmark(
    train_path: str,
    test_path: str,
    output_path: str,
    train_rows: int = 12000,
    evaluation_rows: int = 500,
    epochs: int = 40,
    seed: int = 42,
) -> pd.DataFrame:
    train = pd.read_parquet(train_path).sort_values("timestamp").reset_index(drop=True)
    competition_test = pd.read_parquet(test_path)
    train_iv_columns, _, _ = _iv_columns(train.columns)
    test_iv_columns, _, _ = _iv_columns(competition_test.columns)
    test_column_set = set(test_iv_columns)
    iv_cols = [column for column in train_iv_columns if column in test_column_set]
    if not iv_cols:
        raise ValueError("Train and test do not share any IV strike columns.")
    iv_cols, strikes, call_side = _iv_columns(iv_cols)
    test_only_columns = sorted(test_column_set - set(train_iv_columns))
    train_only_columns = sorted(set(train_iv_columns) - test_column_set)

    train_indices, val_indices, test_indices = split_indices_by_timestamp(
        train["timestamp"].to_numpy(), 0.70, 0.15, seed, temporal=True
    )
    rng = np.random.default_rng(seed)
    train_indices = rng.choice(train_indices, size=min(train_rows, len(train_indices)), replace=False)
    val_indices = np.linspace(
        0, len(val_indices) - 1, num=min(evaluation_rows, len(val_indices)), dtype=int
    )
    test_indices = np.linspace(
        0, len(test_indices) - 1, num=min(evaluation_rows, len(test_indices)), dtype=int
    )
    _, val_full_indices, test_full_indices = split_indices_by_timestamp(
        train["timestamp"].to_numpy(), 0.70, 0.15, seed, temporal=True
    )
    val_frame = train.iloc[val_full_indices[val_indices]].reset_index(drop=True)
    test_frame = train.iloc[test_full_indices[test_indices]].reset_index(drop=True)
    train_frame = train.iloc[train_indices].reset_index(drop=True)

    test_counts = competition_test[iv_cols].notna().to_numpy()
    side_counts = np.column_stack([
        test_counts[:, call_side].sum(axis=1),
        test_counts[:, ~call_side].sum(axis=1),
    ])
    pattern_rng = np.random.default_rng(seed + 1)

    train_values, train_coordinates = _surface_features(train_frame, strikes, iv_cols)
    val_values, val_coordinates = _surface_features(val_frame, strikes, iv_cols)
    test_values, test_coordinates = _surface_features(test_frame, strikes, iv_cols)
    train_counts = side_counts[pattern_rng.integers(0, len(side_counts), size=len(train_frame))]
    val_counts = side_counts[pattern_rng.integers(0, len(side_counts), size=len(val_frame))]
    holdout_counts = side_counts[pattern_rng.integers(0, len(side_counts), size=len(test_frame))]
    train_masks, _ = _sample_masks(train_values, call_side, train_counts, pattern_rng)
    val_masks, val_hidden = _sample_masks(val_values, call_side, val_counts, pattern_rng)
    holdout_masks, holdout_hidden = _sample_masks(test_values, call_side, holdout_counts, pattern_rng)

    mlp = _fit_surface_mlp(
        train_values, train_coordinates, train_masks, epochs=epochs, seed=seed
    )
    pinn = _fit_pinn(
        train_values, train_coordinates, train_masks, epochs=epochs, seed=seed
    )
    results = []
    holdout_predictions = {}
    for split_name, frame, values, masks, hidden, coordinates in (
        ("validation", val_frame, val_values, val_masks, val_hidden, val_coordinates),
        ("holdout", test_frame, test_values, holdout_masks, holdout_hidden, test_coordinates),
    ):
        for method in MODEL_NAMES:
            failures = 0
            if method in ("surface_mlp", "pinn"):
                model_input = _mlp_inputs(values, masks, coordinates).astype(np.float32)
                model = mlp if method == "surface_mlp" else pinn
                model.eval()
                with torch.no_grad():
                    predictions = model(torch.from_numpy(model_input)).numpy()
            else:
                predictions, failures = _evaluate_classical(
                    method, frame, values, masks, strikes, call_side, hidden
                )
            metrics = _score(values, predictions, hidden)
            if split_name == "holdout":
                holdout_predictions[method] = predictions
            results.append({
                "split": split_name,
                "method": method,
                "scored_strikes": len(iv_cols),
                "test_only_strikes_unscored": len(test_only_columns),
                "train_only_strikes_excluded": len(train_only_columns),
                "surfaces": len(frame),
                "observed_quotes_per_surface_median": float(np.median(masks.sum(axis=1))),
                "sabr_failures": failures,
                **metrics,
            })

    result_frame = pd.DataFrame(results)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result_frame.to_csv(output, index=False)
    _plot_holdout_reconstruction(
        test_frame,
        test_values,
        holdout_masks,
        strikes,
        call_side,
        holdout_predictions,
        output.with_name("masked_quote_reconstruction.png"),
    )
    return result_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/nk-iv-prediction/train_data.parquet")
    parser.add_argument("--test", default="data/nk-iv-prediction/test_data.parquet")
    parser.add_argument("--output", default="results/masked_quote_benchmark.csv")
    parser.add_argument("--train-rows", type=int, default=12000)
    parser.add_argument("--evaluation-rows", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    results = run_benchmark(
        args.train,
        args.test,
        args.output,
        train_rows=args.train_rows,
        evaluation_rows=args.evaluation_rows,
        epochs=args.epochs,
        seed=args.seed,
    )
    print(results.to_string(index=False))
    print(f"\nSaved masked-quote benchmark to {args.output}")


if __name__ == "__main__":
    main()
