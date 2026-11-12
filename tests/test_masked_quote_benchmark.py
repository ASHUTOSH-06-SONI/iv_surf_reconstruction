import numpy as np
import torch

from src.evaluate.masked_quote_benchmark import (
    _fit_pinn,
    _fit_sabr,
    _hagan_sabr,
    _interpolate_surface,
    _mlp_inputs,
    _score,
)


def test_linear_interpolation_fills_only_missing_points_by_option_side():
    values = np.array([0.3, 0.2, 0.4, 0.25])
    observed = np.array([True, False, True, False])
    strikes = np.array([90.0, 100.0, 110.0, 120.0])
    call_side = np.array([True, True, False, False])

    predicted = _interpolate_surface(values, observed, strikes, call_side)

    assert np.isnan(predicted[0])
    assert np.isnan(predicted[2])
    assert predicted[1] == 0.3
    assert predicted[3] == 0.4


def test_sabr_fit_recovers_synthetic_smile():
    forward = 100.0
    strikes = np.array([80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0])
    market_ivs = _hagan_sabr(forward, strikes, 0.2, 1.0, 0.7, -0.2, 0.5)

    alpha, rho, nu = _fit_sabr(forward, strikes, market_ivs, 0.2)
    fitted = _hagan_sabr(forward, strikes, 0.2, alpha, 0.7, rho, nu)

    np.testing.assert_allclose(fitted, market_ivs, atol=1e-4)


def test_score_reports_prediction_coverage():
    truth = np.array([[0.2, 0.3, 0.4]])
    predictions = np.array([[np.nan, 0.25, 0.35]])
    hidden = np.array([[True, True, False]])

    scores = _score(truth, predictions, hidden)

    assert scores["n_hidden"] == 2
    assert scores["n_scored"] == 1
    assert scores["coverage"] == 0.5


def test_repository_pinn_trains_on_masked_surface_shapes():
    values = np.array([
        [0.2, 0.3, 0.4, 0.5],
        [0.25, 0.35, 0.45, 0.55],
    ])
    masks = np.array([
        [True, False, True, False],
        [False, True, False, True],
    ])
    coordinates = np.zeros((2, 4))

    model = _fit_pinn(values, coordinates, masks, epochs=1, seed=3)
    inputs = _mlp_inputs(values, masks, coordinates).astype(np.float32)
    predictions = model(torch.from_numpy(inputs))

    assert predictions.shape == values.shape
    assert torch.all(predictions > 0)


def test_pinn_trains_with_parity_and_curvature_penalties():
    from src.models.surface_physics import SurfaceGeometry

    strikes = np.array([90.0, 100.0, 110.0, 90.0, 100.0, 110.0])
    call_side = np.array([True, True, True, False, False, False])
    rng = np.random.default_rng(0)
    values = 0.2 + 0.01 * rng.random((8, 6))
    masks = rng.random((8, 6)) < 0.5
    coordinates = np.zeros((8, 2))
    geometry = SurfaceGeometry.from_columns(strikes, call_side)

    model = _fit_pinn(
        values, coordinates, masks, epochs=2, seed=1,
        parity_weight=1.0, curvature_weight=1.0, curvature_mode="convex", geometry=geometry,
    )
    inputs = _mlp_inputs(values, masks, coordinates).astype(np.float32)
    assert torch.isfinite(model(torch.from_numpy(inputs))).all()


def test_pinn_penalty_without_geometry_is_rejected():
    import pytest

    values = np.full((2, 4), 0.2)
    masks = np.array([[True, False, True, False]] * 2)
    with pytest.raises(ValueError):
        _fit_pinn(values, np.zeros((2, 4)), masks, epochs=1, seed=0, parity_weight=1.0)
