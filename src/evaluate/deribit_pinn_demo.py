"""Chronological masked-quote demonstration on collected Deribit ETH surfaces.

This is a pipeline check for a short live-data capture, not a statistically
meaningful model evaluation.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluate.masked_quote_benchmark import _fit_pinn, _mlp_inputs, _score


def _surface_arrays(
    frame: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Build a shared call/put strike grid across all snapshot-expiry surfaces."""
    required = {
        "snapshot_timestamp_ms",
        "expiration_timestamp",
        "time_to_expiry_seconds",
        "strike",
        "call_iv",
        "put_iv",
        "call_quote_age_seconds",
        "put_quote_age_seconds",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Surface data is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Surface data contains no rows.")

    data = frame.copy()
    data["strike"] = pd.to_numeric(data["strike"], errors="raise")
    strikes = np.sort(data["strike"].unique())
    strike_to_slot = {float(strike): index for index, strike in enumerate(strikes)}
    grouped = list(
        data.groupby(["snapshot_timestamp_ms", "expiration_timestamp"], sort=True)
    )
    values = np.full((len(grouped), 2 * len(strikes)), np.nan, dtype=np.float64)
    ages = np.full_like(values, np.nan)
    metadata_rows: List[Dict[str, object]] = []
    for row_index, ((snapshot_ms, expiry_ms), surface) in enumerate(grouped):
        first = surface.iloc[0]
        for item in surface.itertuples(index=False):
            slot = strike_to_slot[float(item.strike)]
            values[row_index, slot] = item.call_iv
            values[row_index, len(strikes) + slot] = item.put_iv
            ages[row_index, slot] = item.call_quote_age_seconds
            ages[row_index, len(strikes) + slot] = item.put_quote_age_seconds
        metadata_rows.append(
            {
                "surface_id": str(first.get("surface_id", f"{snapshot_ms}_{expiry_ms}")),
                "snapshot_timestamp_ms": int(snapshot_ms),
                "expiration_timestamp": int(expiry_ms),
                "snapshot_time_utc": first.get("snapshot_time_utc"),
                "expiration_time_utc": first.get("expiration_time_utc"),
                "time_to_expiry_seconds": float(first["time_to_expiry_seconds"]),
                "row_index": row_index,
            }
        )
    metadata = pd.DataFrame(metadata_rows)
    return metadata, strikes, values, ages


def _sample_observation_masks(
    values: np.ndarray, mask_fraction: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    if not 0.0 < mask_fraction < 1.0:
        raise ValueError("mask_fraction must be strictly between 0 and 1.")
    rng = np.random.default_rng(seed)
    observed = np.zeros(values.shape, dtype=bool)
    for row_index, row in enumerate(values):
        for start, stop in ((0, values.shape[1] // 2), (values.shape[1] // 2, values.shape[1])):
            valid = np.flatnonzero(np.isfinite(row[start:stop])) + start
            if not len(valid):
                continue
            count = min(len(valid), max(1, int(round(len(valid) * (1.0 - mask_fraction)))))
            observed[row_index, rng.choice(valid, count, replace=False)] = True
    hidden = (~observed) & np.isfinite(values)
    return observed, hidden


def _interpolate_row(
    values: np.ndarray, observed: np.ndarray, strikes: np.ndarray, pooled: bool
) -> np.ndarray:
    predictions = np.full(values.shape, np.nan, dtype=np.float64)
    n_strikes = len(strikes)
    for side_start in (0, n_strikes):
        side_slice = slice(side_start, side_start + n_strikes)
        side_values = values[side_slice]
        known = observed[side_slice] & np.isfinite(side_values)
        if pooled:
            opposite_start = n_strikes if side_start == 0 else 0
            opposite_values = values[opposite_start : opposite_start + n_strikes]
            opposite_known = observed[opposite_start : opposite_start + n_strikes] & np.isfinite(
                opposite_values
            )
            points = np.concatenate([strikes[known], strikes[opposite_known]])
            quotes = np.concatenate([side_values[known], opposite_values[opposite_known]])
        else:
            points = strikes[known]
            quotes = side_values[known]
        if not len(points):
            continue
        unique_points, inverse = np.unique(points, return_inverse=True)
        pooled_quotes = np.zeros(len(unique_points), dtype=np.float64)
        counts = np.zeros(len(unique_points), dtype=np.int64)
        np.add.at(pooled_quotes, inverse, quotes)
        np.add.at(counts, inverse, 1)
        knots = pooled_quotes / counts
        predictions[side_slice] = np.interp(strikes, unique_points, knots)
    return predictions


def _chronological_split(metadata: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps = np.sort(metadata["snapshot_timestamp_ms"].unique())
    if len(timestamps) < 4:
        raise ValueError("At least four distinct snapshot timestamps are required.")
    train_end = max(1, int(len(timestamps) * 0.70))
    validation_end = max(train_end + 1, int(len(timestamps) * 0.85))
    validation_end = min(validation_end, len(timestamps) - 1)
    train_times = set(timestamps[:train_end])
    validation_times = set(timestamps[train_end:validation_end])
    holdout_times = set(timestamps[validation_end:])
    train_indices = np.flatnonzero(metadata["snapshot_timestamp_ms"].isin(train_times))
    validation_indices = np.flatnonzero(metadata["snapshot_timestamp_ms"].isin(validation_times))
    holdout_indices = np.flatnonzero(metadata["snapshot_timestamp_ms"].isin(holdout_times))
    if not len(train_indices) or not len(validation_indices) or not len(holdout_indices):
        raise ValueError("Chronological split produced an empty partition.")
    return train_indices, validation_indices, holdout_indices


def _coordinates(strikes: np.ndarray, values: np.ndarray, metadata: pd.DataFrame) -> np.ndarray:
    """Encode normalized log-strike and log time-to-expiry per quote.

    These are strike coordinates, not true log-moneyness: the capture lacks a
    synchronized ETH index or forward price.
    """
    strike_axis = np.log(strikes / np.median(strikes))
    quote_strikes = np.tile(strike_axis, 2)
    n_quotes = values.shape[1]
    log_maturity = np.log1p(
        np.maximum(metadata["time_to_expiry_seconds"].to_numpy(), 0.0) / 86400.0
    )
    coordinates = np.empty((len(metadata), 2 * n_quotes), dtype=np.float64)
    coordinates[:, 0::2] = quote_strikes[None, :]
    coordinates[:, 1::2] = log_maturity[:, None]
    return coordinates


def run_demo(
    input_path: Path,
    output_path: Path,
    epochs: int = 100,
    mask_fraction: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    surface_frame = pd.read_csv(input_path)
    metadata, strikes, values, ages = _surface_arrays(surface_frame)
    train_indices, validation_indices, holdout_indices = _chronological_split(metadata)
    observed, hidden = _sample_observation_masks(values, mask_fraction, seed)
    coordinates = _coordinates(strikes, values, metadata)

    model = _fit_pinn(
        values[train_indices],
        coordinates[train_indices],
        observed[train_indices],
        epochs=epochs,
        seed=seed,
        range_weight=0.1,
    )
    import torch

    model.eval()
    with torch.no_grad():
        predictions = model(
            torch.from_numpy(
                _mlp_inputs(values, observed, coordinates).astype(np.float32)
            )
        ).numpy()

    methods = {
        "linear_interpolation": np.vstack(
            [
                _interpolate_row(values[i], observed[i], strikes, pooled=False)
                for i in range(len(values))
            ]
        ),
        "merged_interpolation": np.vstack(
            [
                _interpolate_row(values[i], observed[i], strikes, pooled=True)
                for i in range(len(values))
            ]
        ),
        "pinn": predictions,
    }
    split_indices = {"validation": validation_indices, "holdout": holdout_indices}
    summary_rows = []
    prediction_rows = []
    for split, indices in split_indices.items():
        split_hidden = hidden[indices]
        for method, method_predictions in methods.items():
            scores = _score(values[indices], method_predictions[indices], split_hidden)
            summary_rows.append(
                {
                    "split": split,
                    "method": method,
                    "mae": scores["mae"],
                    "rmse": scores["rmse"],
                    "r2": scores["r2"],
                    "n_hidden": scores["n_hidden"],
                    "n_scored": scores["n_scored"],
                    "coverage": scores["coverage"],
                }
            )
        for row_index in indices:
            meta = metadata.iloc[row_index]
            n_strikes = len(strikes)
            for quote_index in np.flatnonzero(hidden[row_index]):
                side = "call" if quote_index < n_strikes else "put"
                strike_index = quote_index % n_strikes
                quote = {
                    "split": split,
                    "surface_id": meta["surface_id"],
                    "snapshot_time_utc": meta["snapshot_time_utc"],
                    "expiration_time_utc": meta["expiration_time_utc"],
                    "strike": strikes[strike_index],
                    "option_type": side,
                    "true_iv": values[row_index, quote_index],
                    "quote_age_seconds": ages[row_index, quote_index],
                }
                for method, method_predictions in methods.items():
                    quote[f"{method}_iv"] = method_predictions[row_index, quote_index]
                prediction_rows.append(quote)

    summary = pd.DataFrame(summary_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path.with_name(output_path.stem + "_summary.csv"), index=False)
    pd.DataFrame(prediction_rows).to_csv(
        output_path.with_name(output_path.stem + "_predictions.csv"), index=False
    )
    _plot_holdout(
        metadata,
        strikes,
        values,
        observed,
        hidden,
        methods,
        holdout_indices,
        output_path.with_suffix(".png"),
    )
    print(
        f"Snapshots: {metadata['snapshot_timestamp_ms'].nunique()} | "
        f"surfaces: {len(metadata)} | train/validation/holdout surfaces: "
        f"{len(train_indices)}/{len(validation_indices)}/{len(holdout_indices)}"
    )
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"Saved metrics to {output_path.with_name(output_path.stem + '_summary.csv')}")
    print(
        "Saved held-out predictions to "
        f"{output_path.with_name(output_path.stem + '_predictions.csv')} and plot to "
        f"{output_path.with_suffix('.png')}"
    )
    return summary


def _plot_holdout(
    metadata: pd.DataFrame,
    strikes: np.ndarray,
    values: np.ndarray,
    observed: np.ndarray,
    hidden: np.ndarray,
    methods: Mapping[str, np.ndarray],
    holdout_indices: np.ndarray,
    output_path: Path,
) -> None:
    if not len(holdout_indices):
        return
    representative = metadata.iloc[holdout_indices].sort_values(
        ["snapshot_timestamp_ms", "time_to_expiry_seconds"]
    )
    row_index = int(representative.index[0])
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    n_strikes = len(strikes)
    colors = {
        "linear_interpolation": "tab:blue",
        "merged_interpolation": "tab:orange",
        "pinn": "tab:purple",
    }
    for side_index, (side, axis) in enumerate(zip(("call", "put"), axes)):
        start = side_index * n_strikes
        indices = np.arange(start, start + n_strikes)
        side_observed = observed[row_index, indices]
        side_hidden = hidden[row_index, indices]
        axis.scatter(
            strikes[side_observed],
            values[row_index, indices][side_observed],
            color="black",
            marker="o",
            label="Observed input",
            zorder=4,
        )
        axis.scatter(
            strikes[side_hidden],
            values[row_index, indices][side_hidden],
            color="red",
            marker="x",
            label="Hidden target",
            zorder=4,
        )
        for method, predictions in methods.items():
            axis.plot(
                strikes[side_hidden],
                predictions[row_index, indices][side_hidden],
                marker=".",
                color=colors[method],
                linewidth=1.2,
                label=method.replace("_", " "),
            )
        axis.set_title(f"{side.title()} IV")
        axis.set_xlabel("Strike")
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("Deribit mark IV (raw units)")
    axes[1].legend(fontsize=8)
    meta = metadata.iloc[row_index]
    figure.suptitle(
        "ETH option surface masked-quote demo\n"
        f"{meta['snapshot_time_utc']} | expiry {meta['expiration_time_utc']}"
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("data/processed/deribit/eth_iv_surfaces.csv")
    )
    parser.add_argument("--output", type=Path, default=Path("results/deribit_pinn_demo.csv"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--mask-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 0:
        parser.error("--epochs must be non-negative.")
    run_demo(args.input, args.output, args.epochs, args.mask_fraction, args.seed)


if __name__ == "__main__":
    main()
