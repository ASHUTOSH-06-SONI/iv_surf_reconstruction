"""Benchmark IV quote reconstruction from partially observed historical surfaces."""

import argparse
import logging
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from torch.utils.data import DataLoader, TensorDataset

from src.models import (
    PINNModel,
    ResidualSurfaceModel,
    SurfaceGeometry,
    curvature_penalty,
    parity_penalty,
    physics_penalty,
)
from src.models.surface_physics import CURVATURE_MODES
from src.train.train_and_save import split_indices_by_timestamp


LOGGER = logging.getLogger(__name__)
MODEL_NAMES = (
    "surface_median",
    "linear_interpolation",
    "parity_copy",
    "merged_interpolation",
    "merged_linear_wing",
    "merged_quad_wing",
    "sabr",
    "surface_mlp",
    "pinn",
    "pinn_residual",
)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


def load_physics_settings(config_path: Optional[str] = None) -> Dict[str, object]:
    """Read physics/ablation settings from the YAML config (no directories are created)."""
    with open(config_path or DEFAULT_CONFIG_PATH, "r") as handle:
        raw = yaml.safe_load(handle) or {}
    physics = raw.get("physics", {})
    return {
        "range_weight": float(raw.get("model", {}).get("physics_penalty_weight", 0.1)),
        "parity_weight": float(physics.get("parity_weight", 0.0)),
        "curvature_weight": float(physics.get("curvature_weight", 0.0)),
        "curvature_mode": str(physics.get("curvature_mode", "smooth")),
        "ablation": raw.get("ablation", {}),
    }


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


def _partner_index(strikes: np.ndarray, is_call: np.ndarray) -> np.ndarray:
    """Column index of the same-strike, opposite-side quote (-1 if there is none)."""
    partner = np.full(len(strikes), -1, dtype=np.int64)
    at = {(float(k), bool(c)): i for i, (k, c) in enumerate(zip(strikes, is_call))}
    for (strike, call_side), index in at.items():
        partner[index] = at.get((strike, not call_side), -1)
    return partner


def _parity_copy(
    values: np.ndarray,
    observed: np.ndarray,
    strikes: np.ndarray,
    is_call: np.ndarray,
    partner: np.ndarray,
) -> np.ndarray:
    """Copy the observed opposite-side quote at the same strike; else per-side interpolation.

    Isolates what put-call parity alone buys: a hidden quote whose counterpart is
    visible gets that value, everything else falls back to ``_interpolate_surface``.
    """
    prediction = _interpolate_surface(values, observed, strikes, is_call)
    safe = np.where(partner >= 0, partner, 0)
    copyable = (partner >= 0) & observed[safe] & np.isfinite(values[safe]) & ~observed
    prediction[copyable] = values[safe][copyable]
    return prediction


def _merged_interpolation(
    values: np.ndarray,
    observed: np.ndarray,
    strikes: np.ndarray,
    unique_strikes: np.ndarray,
    strike_slot: np.ndarray,
) -> np.ndarray:
    """Interpolate one pooled smile per surface, then read it off for both sides.

    Calls and puts share an IV at each strike, so their observed quotes are pooled
    (averaged when both are visible) onto a single per-strike curve. ``np.interp``
    is exact at its knots, so a hidden quote with an observed counterpart receives
    that counterpart's value, and the remaining strikes are interpolated using
    information from both sides.
    """
    prediction = np.full(len(values), np.nan, dtype=np.float64)
    known = observed & np.isfinite(values)
    target = ~observed
    if not known.any() or not target.any():
        return prediction
    total = np.bincount(strike_slot[known], weights=values[known], minlength=len(unique_strikes))
    count = np.bincount(strike_slot[known], minlength=len(unique_strikes))
    has = count > 0
    curve = total[has] / count[has]
    prediction[target] = np.interp(strikes[target], unique_strikes[has], curve)
    return prediction


WING_MODES = ("linear", "quadratic")
WING_FIT_KNOTS = 4        # outermost pooled knots used for the linear wing fit
IV_FLOOR, IV_CEIL = 0.01, 3.0


def _merged_wing(
    values: np.ndarray,
    observed: np.ndarray,
    strikes: np.ndarray,
    unique_strikes: np.ndarray,
    strike_slot: np.ndarray,
    underlying: float,
    mode: str,
) -> np.ndarray:
    """``merged_interpolation`` inside the observed strike range, a fitted wing outside it.

    Quotes are pooled per strike as in ``_merged_interpolation``. Hidden quotes
    beyond the outermost observed strike are extrapolated in log-moneyness:

    * ``"linear"``: least-squares line through the outermost ``WING_FIT_KNOTS``
      pooled knots on that side, anchored at the edge knot.
    * ``"quadratic"``: one least-squares parabola through all pooled knots (needs 3),
      applied as an increment from the edge knot so the surface stays continuous.

    Falls back to the flat edge value when there are too few knots.
    """
    if mode not in WING_MODES:
        raise ValueError(f"Unknown wing mode '{mode}'. Options: {WING_MODES}")
    prediction = _merged_interpolation(values, observed, strikes, unique_strikes, strike_slot)
    known = observed & np.isfinite(values)
    target = ~observed
    if not known.any() or not target.any():
        return prediction
    total = np.bincount(strike_slot[known], weights=values[known], minlength=len(unique_strikes))
    count = np.bincount(strike_slot[known], minlength=len(unique_strikes))
    has = count > 0
    knot_x = np.log(unique_strikes[has] / underlying)
    knot_y = total[has] / count[has]
    if len(knot_x) < 2:
        return prediction
    log_k = np.log(strikes / underlying)
    quad = np.polyfit(knot_x, knot_y, 2) if mode == "quadratic" and len(knot_x) >= 3 else None
    for side, edge in (("low", 0), ("high", -1)):
        outside = target & (log_k < knot_x[0] if side == "low" else log_k > knot_x[-1])
        if not outside.any():
            continue
        x_edge, y_edge = knot_x[edge], knot_y[edge]
        if quad is not None:
            delta = np.polyval(quad, log_k[outside]) - np.polyval(quad, x_edge)
        else:
            n = min(WING_FIT_KNOTS, len(knot_x))
            sel = slice(0, n) if side == "low" else slice(-n, None)
            slope = np.polyfit(knot_x[sel], knot_y[sel], 1)[0] if n >= 2 else 0.0
            delta = slope * (log_k[outside] - x_edge)
        prediction[outside] = np.clip(y_edge + delta, IV_FLOOR, IV_CEIL)
    return prediction


def parity_gap_report(
    frame: pd.DataFrame, iv_cols: Sequence[str], strikes: np.ndarray, is_call: np.ndarray
) -> Dict[str, float]:
    """Distribution of |call IV - put IV| over same-strike pairs where both are present.

    Needs no masking, so it also works directly on the partially observed
    competition test file and shows whether pooling calls and puts is safe.
    """
    values = frame[list(iv_cols)].to_numpy(dtype=np.float64, copy=True)
    values[(values < 0.0) | (values > 3.0)] = np.nan
    partner = _partner_index(strikes, is_call)
    call_index = np.flatnonzero(is_call & (partner >= 0))
    gap = np.abs(values[:, call_index] - values[:, partner[call_index]])
    gap = gap[np.isfinite(gap)]
    if gap.size == 0:
        return {"n_pairs": 0}
    return {
        "n_pairs": int(gap.size),
        "rms": float(np.sqrt(np.mean(gap**2))),
        "median_abs": float(np.median(gap)),
        "p95_abs": float(np.quantile(gap, 0.95)),
        "p99_abs": float(np.quantile(gap, 0.99)),
        "max_abs": float(gap.max()),
    }


def _counterpart_observed_fraction(
    observed: np.ndarray, hidden: np.ndarray, partner: np.ndarray
) -> float:
    """Share of hidden quotes whose same-strike opposite-side quote is observed."""
    safe = np.where(partner >= 0, partner, 0)
    available = observed[:, safe] & (partner >= 0)[None, :]
    n_hidden = hidden.sum()
    return float((hidden & available).sum() / n_hidden) if n_hidden else float("nan")


def _surface_features(
    frame: pd.DataFrame,
    strikes: np.ndarray,
    iv_cols: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    values = frame[list(iv_cols)].to_numpy(dtype=np.float64, copy=True)
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


def _merged_base(
    values: np.ndarray, observed: np.ndarray, strikes: np.ndarray, fill_value: float
) -> np.ndarray:
    """Completed surfaces from ``merged_interpolation``: observed quotes kept, hidden filled.

    Rows with nothing observed (and any other non-finite entries) get ``fill_value``.
    """
    unique_strikes, strike_slot = np.unique(strikes, return_inverse=True)
    base = np.empty(values.shape, dtype=np.float64)
    for row in range(len(values)):
        filled = _merged_interpolation(
            values[row], observed[row], strikes, unique_strikes, strike_slot
        )
        base[row] = np.where(observed[row], values[row], filled)
    return np.where(np.isfinite(base), base, fill_value)


def _residual_inputs(
    values: np.ndarray,
    observed: np.ndarray,
    coordinates: np.ndarray,
    strikes: np.ndarray,
    fill_value: float,
) -> np.ndarray:
    """Model inputs plus the merged base surface as trailing columns."""
    base = _merged_base(values, observed, strikes, fill_value)
    return np.concatenate([_mlp_inputs(values, observed, coordinates), base], axis=1).astype(
        np.float32
    )


def _fit_pinn(
    train_values: np.ndarray,
    train_coordinates: np.ndarray,
    train_masks: np.ndarray,
    epochs: int,
    seed: int,
    range_weight: float = 0.1,
    parity_weight: float = 0.0,
    curvature_weight: float = 0.0,
    curvature_mode: str = "smooth",
    geometry: Optional[SurfaceGeometry] = None,
    residual_base: Optional[np.ndarray] = None,
) -> nn.Module:
    """Train the repository PINN on masked surfaces with physics penalties.

    ``range_weight`` scales the IV-range penalty on raw predictions. The parity
    and curvature penalties act on the *completed* surface (observed quotes
    kept, hidden quotes replaced by predictions), which is the surface a user
    would actually receive. They need ``geometry`` and are skipped at weight 0.

    With ``residual_base`` (completed base surfaces, same shape as ``train_values``)
    the network predicts a correction added to that base and starts exactly at it.
    """
    if (parity_weight > 0 or curvature_weight > 0) and geometry is None:
        raise ValueError("parity/curvature penalties require a SurfaceGeometry.")
    torch.manual_seed(seed)
    inputs = _mlp_inputs(train_values, train_masks, train_coordinates).astype(np.float32)
    if residual_base is not None:
        inputs = np.concatenate([inputs, residual_base.astype(np.float32)], axis=1)
    targets = np.nan_to_num(train_values, nan=0.0).astype(np.float32)
    hidden = (~train_masks) & np.isfinite(train_values)
    n_quotes = train_values.shape[1]
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
        output_dim=n_quotes,
        output_activation="softplus" if residual_base is None else "none",
    )
    if residual_base is not None:
        model = ResidualSurfaceModel(model, n_quotes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1.0e-5)
    for _ in range(epochs):
        model.train()
        for batch_inputs, batch_targets, batch_hidden in loader:
            optimizer.zero_grad()
            predictions = model(batch_inputs)
            supervised = ((predictions - batch_targets) ** 2 * batch_hidden).sum()
            supervised = supervised / batch_hidden.sum().clamp_min(1.0)
            loss = supervised
            if range_weight > 0:
                loss = loss + range_weight * physics_penalty(predictions, iv_min=0.0, iv_max=3.0)
            if parity_weight > 0 or curvature_weight > 0:
                observed = batch_inputs[:, n_quotes : 2 * n_quotes] > 0.5
                completed = torch.where(observed, batch_targets, predictions)
                if parity_weight > 0:
                    loss = loss + parity_weight * parity_penalty(completed, geometry)
                if curvature_weight > 0:
                    loss = loss + curvature_weight * curvature_penalty(
                        completed, geometry, mode=curvature_mode
                    )
            loss.backward()
            optimizer.step()
    return model


def _region_masks(
    values: np.ndarray, observed: np.ndarray, hidden: np.ndarray, strikes: np.ndarray
) -> Dict[str, np.ndarray]:
    """Split hidden quotes into interior vs extrapolated, per surface.

    A hidden quote is *interior* when its strike lies within the range of strikes
    that are observed on that surface (either side pooled), otherwise it must be
    extrapolated. One definition for all methods keeps the comparison on identical
    quotes; a per-side method can still extrapolate on an "interior" quote.
    """
    known = observed & np.isfinite(values)
    low = np.where(known, strikes[None, :], np.inf).min(axis=1, keepdims=True)
    high = np.where(known, strikes[None, :], -np.inf).max(axis=1, keepdims=True)
    inside = (strikes[None, :] >= low) & (strikes[None, :] <= high)
    return {"interior": hidden & inside, "extrapolated": hidden & ~inside}


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
    partner = _partner_index(strikes, call_side)
    unique_strikes, strike_slot = np.unique(strikes, return_inverse=True)
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
        elif method == "parity_copy":
            predictions[row] = _parity_copy(
                values[row], observed[row], strikes, call_side, partner
            )
        elif method == "merged_interpolation":
            predictions[row] = _merged_interpolation(
                values[row], observed[row], strikes, unique_strikes, strike_slot
            )
        elif method in ("merged_linear_wing", "merged_quad_wing"):
            predictions[row] = _merged_wing(
                values[row], observed[row], strikes, unique_strikes, strike_slot,
                float(frame.iloc[row]["underlying"]),
                "linear" if method == "merged_linear_wing" else "quadratic",
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
        "parity_copy": "tab:cyan",
        "pinn_residual": "tab:purple",
        "merged_interpolation": "tab:red",
        "merged_linear_wing": "tab:orange",
        "merged_quad_wing": "tab:brown",
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


SPLIT_MODES = ("temporal", "expiry")


def _split_pools(
    train: pd.DataFrame, seed: int, split_mode: str, holdout_expiry: Optional[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row positions (into the time-sorted frame) for the train / validation / holdout pools.

    ``temporal``: first 70% / next 15% / last 15% of time (all expiries mixed).
    ``expiry``: every row of ``holdout_expiry`` (default: the latest expiry) is unseen
    in training; its first half in time is validation, its second half holdout.
    """
    if split_mode not in SPLIT_MODES:
        raise ValueError(f"Unknown split mode '{split_mode}'. Options: {SPLIT_MODES}")
    if split_mode == "temporal":
        return split_indices_by_timestamp(train["timestamp"].to_numpy(), 0.70, 0.15, seed, temporal=True)
    expiries = train["expiry"].astype(str).to_numpy()
    held = str(holdout_expiry) if holdout_expiry else sorted(set(expiries))[-1]
    held_rows = np.flatnonzero(expiries == held)
    if len(held_rows) < 2 or len(held_rows) == len(train):
        raise ValueError(f"Expiry '{held}' cannot be held out (rows: {len(held_rows)}).")
    half = len(held_rows) // 2
    return np.flatnonzero(expiries != held), held_rows[:half], held_rows[half:]


def _prepare_surfaces(
    train_path: str,
    test_path: str,
    train_rows: int,
    evaluation_rows: int,
    seed: int,
    split_mode: str = "temporal",
    holdout_expiry: Optional[str] = None,
) -> SimpleNamespace:
    """Load data, split it and sample observed/hidden masks (seed-controlled)."""
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

    train_pool, val_pool, test_pool = _split_pools(train, seed, split_mode, holdout_expiry)
    rng = np.random.default_rng(seed)
    train_indices = rng.choice(train_pool, size=min(train_rows, len(train_pool)), replace=False)
    val_pick = np.linspace(0, len(val_pool) - 1, num=min(evaluation_rows, len(val_pool)), dtype=int)
    test_pick = np.linspace(0, len(test_pool) - 1, num=min(evaluation_rows, len(test_pool)), dtype=int)
    val_frame = train.iloc[val_pool[val_pick]].reset_index(drop=True)
    test_frame = train.iloc[test_pool[test_pick]].reset_index(drop=True)
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

    return SimpleNamespace(
        train_frame=train_frame,
        full_train=train,
        competition_test=competition_test,
        iv_cols=iv_cols,
        strikes=strikes,
        call_side=call_side,
        test_only_columns=test_only_columns,
        train_only_columns=train_only_columns,
        val_frame=val_frame,
        test_frame=test_frame,
        train_values=train_values,
        train_coordinates=train_coordinates,
        train_masks=train_masks,
        val_values=val_values,
        val_coordinates=val_coordinates,
        val_masks=val_masks,
        val_hidden=val_hidden,
        test_values=test_values,
        test_coordinates=test_coordinates,
        holdout_masks=holdout_masks,
        holdout_hidden=holdout_hidden,
    )


def run_benchmark(
    train_path: str,
    test_path: str,
    output_path: str,
    train_rows: int = 12000,
    evaluation_rows: int = 500,
    epochs: int = 40,
    seed: int = 42,
    range_weight: float = 0.1,
    parity_weight: float = 0.0,
    curvature_weight: float = 0.0,
    curvature_mode: str = "smooth",
    split_mode: str = "temporal",
    holdout_expiry: Optional[str] = None,
) -> pd.DataFrame:
    data = _prepare_surfaces(
        train_path, test_path, train_rows, evaluation_rows, seed, split_mode, holdout_expiry
    )
    iv_cols, strikes, call_side = data.iv_cols, data.strikes, data.call_side
    test_only_columns, train_only_columns = data.test_only_columns, data.train_only_columns
    val_frame, test_frame = data.val_frame, data.test_frame
    train_values, train_coordinates, train_masks = (
        data.train_values, data.train_coordinates, data.train_masks,
    )
    val_values, val_coordinates, val_masks, val_hidden = (
        data.val_values, data.val_coordinates, data.val_masks, data.val_hidden,
    )
    test_values, test_coordinates = data.test_values, data.test_coordinates
    holdout_masks, holdout_hidden = data.holdout_masks, data.holdout_hidden
    partner = _partner_index(strikes, call_side)

    mlp = _fit_surface_mlp(
        train_values, train_coordinates, train_masks, epochs=epochs, seed=seed
    )
    pinn = _fit_pinn(
        train_values,
        train_coordinates,
        train_masks,
        epochs=epochs,
        seed=seed,
        range_weight=range_weight,
        parity_weight=parity_weight,
        curvature_weight=curvature_weight,
        curvature_mode=curvature_mode,
        geometry=SurfaceGeometry.from_columns(strikes, call_side),
    )
    fill_value = float(np.nanmedian(train_values))
    residual = _fit_pinn(
        train_values,
        train_coordinates,
        train_masks,
        epochs=epochs,
        seed=seed,
        range_weight=range_weight,
        parity_weight=parity_weight,
        curvature_weight=curvature_weight,
        curvature_mode=curvature_mode,
        geometry=SurfaceGeometry.from_columns(strikes, call_side),
        residual_base=_merged_base(train_values, train_masks, strikes, fill_value),
    )
    results = []
    region_rows = []
    holdout_predictions = {}
    for split_name, frame, values, masks, hidden, coordinates in (
        ("validation", val_frame, val_values, val_masks, val_hidden, val_coordinates),
        ("holdout", test_frame, test_values, holdout_masks, holdout_hidden, test_coordinates),
    ):
        for method in MODEL_NAMES:
            failures = 0
            if method in ("surface_mlp", "pinn", "pinn_residual"):
                if method == "pinn_residual":
                    model_input = _residual_inputs(values, masks, coordinates, strikes, fill_value)
                else:
                    model_input = _mlp_inputs(values, masks, coordinates).astype(np.float32)
                model = {"surface_mlp": mlp, "pinn": pinn, "pinn_residual": residual}[method]
                model.eval()
                with torch.no_grad():
                    predictions = model(torch.from_numpy(model_input)).numpy()
            else:
                predictions, failures = _evaluate_classical(
                    method, frame, values, masks, strikes, call_side, hidden
                )
            metrics = _score(values, predictions, hidden)
            for region, region_hidden in _region_masks(values, masks, hidden, strikes).items():
                region_rows.append({
                    "split": split_name,
                    "method": method,
                    "region": region,
                    **_score(values, predictions, region_hidden),
                })
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
                "counterpart_observed": _counterpart_observed_fraction(masks, hidden, partner),
                **metrics,
            })

    result_frame = pd.DataFrame(results)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result_frame.to_csv(output, index=False)
    pd.DataFrame(region_rows).to_csv(output.with_name("region_breakdown.csv"), index=False)
    diagnostic = pd.DataFrame([
        {"source": name, **parity_gap_report(source, iv_cols, strikes, call_side)}
        for name, source in (
            ("train", data.full_train),
            ("competition_test", data.competition_test),
        )
    ])
    diagnostic.to_csv(output.with_name("parity_diagnostic.csv"), index=False)
    LOGGER.info("Call-put IV gap (|call-put| at the same strike):\n%s", diagnostic.to_string(index=False))
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
    parser.add_argument("--config", default=None, help="YAML with model/physics settings.")
    parser.add_argument(
        "--split-mode", choices=SPLIT_MODES, default="temporal",
        help="temporal: last 15%% of time; expiry: hold out one whole expiry (unseen contract).",
    )
    parser.add_argument("--holdout-expiry", default=None, help="Expiry to hold out (default: latest).")
    parser.add_argument("--range-weight", type=float, default=None)
    parser.add_argument("--parity-weight", type=float, default=None)
    parser.add_argument("--curvature-weight", type=float, default=None)
    parser.add_argument("--curvature-mode", choices=CURVATURE_MODES, default=None)
    args = parser.parse_args()
    settings = load_physics_settings(args.config)
    for key in ("range_weight", "parity_weight", "curvature_weight", "curvature_mode"):
        override = getattr(args, key)
        if override is not None:
            settings[key] = override
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    results = run_benchmark(
        args.train,
        args.test,
        args.output,
        train_rows=args.train_rows,
        evaluation_rows=args.evaluation_rows,
        epochs=args.epochs,
        seed=args.seed,
        range_weight=settings["range_weight"],
        parity_weight=settings["parity_weight"],
        curvature_weight=settings["curvature_weight"],
        curvature_mode=settings["curvature_mode"],
        split_mode=args.split_mode,
        holdout_expiry=args.holdout_expiry,
    )
    print(results.to_string(index=False))
    print(f"\nSaved masked-quote benchmark to {args.output}")


if __name__ == "__main__":
    main()
