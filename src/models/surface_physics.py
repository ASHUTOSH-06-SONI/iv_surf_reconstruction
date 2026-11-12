"""Surface-level structural penalties for multi-output IV surface models.

These act on a full surface vector (one column per call/put strike) rather than
on individual predictions, so they can encode relationships *between* quotes:

* **Put-call parity (in IV space):** for the same expiry and strike, a European
  call and put share one implied volatility, so ``iv_call(K) ~= iv_put(K)``.
* **Strike-curvature:** the second derivative of IV w.r.t. strike, computed on
  the non-uniform strike grid. ``"smooth"`` penalises curvature magnitude;
  ``"convex"`` only penalises negative curvature (a cheap proxy that is NOT a
  full butterfly-arbitrage condition, which needs call prices, forward and T).

Strikes are expressed in units of the median strike spacing, so the curvature is
a second difference per grid step squared: its magnitude is in IV units and
comparable to the MSE, independent of index level. Every function accepts either a ``torch.Tensor`` (differentiable,
used for training) or a ``numpy.ndarray`` (NaN-aware, used for diagnostics) of
shape ``(batch, n_quotes)``.
"""

from dataclasses import dataclass
from typing import Dict, Sequence, Union

import numpy as np
import torch

Array = Union[torch.Tensor, np.ndarray]
CURVATURE_MODES = ("smooth", "convex")


@dataclass(frozen=True)
class SurfaceGeometry:
    """Index structure of a surface vector: parity pairs and curvature stencils."""

    strike_scale: float
    parity_call: np.ndarray
    parity_put: np.ndarray
    curv_lo: np.ndarray
    curv_mid: np.ndarray
    curv_hi: np.ndarray
    coef_lo: np.ndarray
    coef_mid: np.ndarray
    coef_hi: np.ndarray

    @classmethod
    def from_columns(cls, strikes: Sequence[float], is_call: Sequence[bool]) -> "SurfaceGeometry":
        strikes = np.asarray(strikes, dtype=np.float64)
        is_call = np.asarray(is_call, dtype=bool)
        if strikes.shape != is_call.shape or strikes.ndim != 1:
            raise ValueError("strikes and is_call must be 1-D arrays of equal length.")

        by_side = {}
        for side in (True, False):
            idx = np.flatnonzero(is_call == side)
            order = idx[np.argsort(strikes[idx], kind="stable")]
            if len(order) > 1 and np.any(np.diff(strikes[order]) <= 0):
                raise ValueError("Duplicate strikes on one option side.")
            by_side[side] = order

        spacings = np.concatenate(
            [np.diff(strikes[order]) for order in by_side.values() if len(order) > 1]
            or [np.array([1.0])]
        )
        scale = float(np.median(spacings))

        call_at = {strikes[i]: i for i in by_side[True]}
        put_at = {strikes[i]: i for i in by_side[False]}
        common = sorted(set(call_at) & set(put_at))
        parity_call = np.array([call_at[k] for k in common], dtype=np.int64)
        parity_put = np.array([put_at[k] for k in common], dtype=np.int64)

        lo, mid, hi, c_lo, c_mid, c_hi = [], [], [], [], [], []
        for order in by_side.values():
            x = strikes[order] / scale
            for j in range(1, len(order) - 1):
                h1, h2 = x[j] - x[j - 1], x[j + 1] - x[j]
                a = 2.0 / (h1 * (h1 + h2))
                c = 2.0 / (h2 * (h1 + h2))
                lo.append(order[j - 1])
                mid.append(order[j])
                hi.append(order[j + 1])
                c_lo.append(a)
                c_mid.append(-(a + c))
                c_hi.append(c)

        as_idx = lambda v: np.asarray(v, dtype=np.int64)
        as_f = lambda v: np.asarray(v, dtype=np.float64)
        return cls(
            scale,
            parity_call, parity_put,
            as_idx(lo), as_idx(mid), as_idx(hi),
            as_f(c_lo), as_f(c_mid), as_f(c_hi),
        )


def _take(values: Array, index: np.ndarray) -> Array:
    if isinstance(values, torch.Tensor):
        return values[:, torch.as_tensor(index, device=values.device)]
    return values[:, index]


def _coef(values: Array, coefficients: np.ndarray) -> Array:
    if isinstance(values, torch.Tensor):
        return torch.as_tensor(coefficients, dtype=values.dtype, device=values.device)
    return coefficients


def parity_gap(values: Array, geometry: SurfaceGeometry) -> Array:
    """``iv_call(K) - iv_put(K)`` for every strike present on both sides."""
    return _take(values, geometry.parity_call) - _take(values, geometry.parity_put)


def strike_curvature(values: Array, geometry: SurfaceGeometry) -> Array:
    """Second derivative of IV w.r.t. normalised strike, per side, per interior strike."""
    return (
        _coef(values, geometry.coef_lo) * _take(values, geometry.curv_lo)
        + _coef(values, geometry.coef_mid) * _take(values, geometry.curv_mid)
        + _coef(values, geometry.coef_hi) * _take(values, geometry.curv_hi)
    )


def _zero(values: torch.Tensor) -> torch.Tensor:
    return values.sum() * 0.0


def parity_penalty(values: torch.Tensor, geometry: SurfaceGeometry) -> torch.Tensor:
    """Mean squared call-put IV gap across shared strikes (differentiable)."""
    if len(geometry.parity_call) == 0:
        return _zero(values)
    return (parity_gap(values, geometry) ** 2).mean()


def curvature_penalty(
    values: torch.Tensor, geometry: SurfaceGeometry, mode: str = "smooth"
) -> torch.Tensor:
    """Mean squared strike-curvature penalty (differentiable)."""
    if mode not in CURVATURE_MODES:
        raise ValueError(f"Unknown curvature mode '{mode}'. Options: {CURVATURE_MODES}")
    if len(geometry.curv_mid) == 0:
        return _zero(values)
    curvature = strike_curvature(values, geometry)
    if mode == "convex":
        curvature = torch.clamp(-curvature, min=0.0)
    return (curvature**2).mean()


def surface_violation_metrics(values: np.ndarray, geometry: SurfaceGeometry) -> Dict[str, float]:
    """NaN-aware RMS parity gap and RMS strike-curvature of (completed) surfaces."""
    out = {"parity_rms": float("nan"), "curvature_rms": float("nan")}
    if len(geometry.parity_call):
        gap = parity_gap(values, geometry)
        if np.isfinite(gap).any():
            out["parity_rms"] = float(np.sqrt(np.nanmean(gap**2)))
    if len(geometry.curv_mid):
        curvature = strike_curvature(values, geometry)
        if np.isfinite(curvature).any():
            out["curvature_rms"] = float(np.sqrt(np.nanmean(curvature**2)))
    return out
