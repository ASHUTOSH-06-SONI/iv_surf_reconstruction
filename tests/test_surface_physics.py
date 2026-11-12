import numpy as np
import pytest
import torch

from src.models.surface_physics import (
    SurfaceGeometry,
    curvature_penalty,
    parity_gap,
    parity_penalty,
    strike_curvature,
    surface_violation_metrics,
)

# columns: calls at 90/100/110/125 (non-uniform), puts at 90/100/110/130
STRIKES = np.array([90.0, 100.0, 110.0, 125.0, 90.0, 100.0, 110.0, 130.0])
IS_CALL = np.array([True] * 4 + [False] * 4)


def geometry():
    return SurfaceGeometry.from_columns(STRIKES, IS_CALL)


def test_parity_pairs_only_shared_strikes():
    g = geometry()
    assert g.parity_call.tolist() == [0, 1, 2]
    assert g.parity_put.tolist() == [4, 5, 6]


def test_parity_zero_when_call_equals_put_and_positive_otherwise():
    g = geometry()
    smile = np.array([0.3, 0.2, 0.25, 0.3])
    same = torch.tensor([[*smile, 0.3, 0.2, 0.25, 0.9]], dtype=torch.float64)
    assert parity_penalty(same, g).item() == pytest.approx(0.0)
    off = same.clone()
    off[0, 5] += 0.1
    assert parity_penalty(off, g).item() == pytest.approx(0.1**2 / 3)


def test_curvature_recovers_quadratic_on_nonuniform_grid():
    g = geometry()
    scale = g.strike_scale
    a = 0.7
    y = a * (STRIKES / scale) ** 2
    curvature = strike_curvature(y[None, :], g)
    np.testing.assert_allclose(curvature, 2 * a, rtol=1e-9)


def test_curvature_zero_for_linear_surface():
    g = geometry()
    y = 0.1 + 0.002 * STRIKES
    assert curvature_penalty(torch.tensor(y[None, :]), g).item() == pytest.approx(0.0, abs=1e-12)


def test_convex_mode_ignores_convex_and_penalises_concave():
    g = geometry()
    x = STRIKES / geometry().strike_scale
    convex = torch.tensor((0.5 * x**2)[None, :])
    concave = torch.tensor((-0.5 * x**2)[None, :])
    assert curvature_penalty(convex, g, mode="convex").item() == pytest.approx(0.0)
    assert curvature_penalty(concave, g, mode="convex").item() > 0.5
    assert curvature_penalty(convex, g, mode="smooth").item() > 0.5


def test_curvature_scale_is_index_level_invariant():
    smile = lambda k: 0.2 + 0.5 * ((k - 100.0) / 10.0) ** 2 * 0.01
    low = SurfaceGeometry.from_columns(STRIKES, IS_CALL)
    high = SurfaceGeometry.from_columns(STRIKES * 250, IS_CALL)
    y = smile(STRIKES)[None, :]
    np.testing.assert_allclose(strike_curvature(y, low), strike_curvature(y, high), rtol=1e-9)
    assert low.strike_scale == pytest.approx(10.0)


def test_unknown_mode_and_duplicate_strikes_rejected():
    with pytest.raises(ValueError):
        curvature_penalty(torch.zeros(1, 8), geometry(), mode="nope")
    with pytest.raises(ValueError):
        SurfaceGeometry.from_columns([100.0, 100.0], [True, True])


def test_penalties_are_differentiable():
    g = geometry()
    pred = torch.rand(4, 8, requires_grad=True)
    (parity_penalty(pred, g) + curvature_penalty(pred, g)).backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0


def test_degenerate_geometry_returns_zero_with_graph():
    g = SurfaceGeometry.from_columns([100.0, 110.0], [True, True])
    pred = torch.rand(2, 2, requires_grad=True)
    loss = parity_penalty(pred, g) + curvature_penalty(pred, g)
    loss.backward()
    assert loss.item() == 0.0


def test_violation_metrics_are_nan_aware():
    g = geometry()
    values = np.array([[0.3, 0.2, 0.25, 0.3, 0.3, np.nan, 0.25, 0.4]])
    metrics = surface_violation_metrics(values, g)
    assert np.isfinite(metrics["parity_rms"]) and metrics["parity_rms"] == pytest.approx(0.0)
    assert np.isfinite(metrics["curvature_rms"])
    all_nan = np.full((1, 8), np.nan)
    assert np.isnan(surface_violation_metrics(all_nan, g)["parity_rms"])


def test_parity_gap_shape_matches_batch():
    g = geometry()
    assert parity_gap(torch.zeros(5, 8), g).shape == (5, 3)
