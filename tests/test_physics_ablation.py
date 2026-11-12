import numpy as np
import pandas as pd
import pytest
import torch

from src.evaluate.masked_quote_benchmark import (
    _fit_pinn,
    _merged_base,
    _residual_inputs,
)
from src.evaluate.physics_ablation import ablation_variants, residual_variants, summarize


def test_variants_toggle_exactly_the_intended_terms():
    v = ablation_variants(range_weight=0.1, parity_weight=2.0, curvature_weight=3.0)
    assert v["pinn_no_physics"] == dict(range_weight=0.0, parity_weight=0.0, curvature_weight=0.0)
    assert v["pinn_range"]["parity_weight"] == 0.0 and v["pinn_range"]["range_weight"] == 0.1
    assert v["pinn_parity"]["parity_weight"] == 2.0 and v["pinn_parity"]["curvature_weight"] == 0.0
    assert v["pinn_curvature"]["curvature_weight"] == 3.0 and v["pinn_curvature"]["parity_weight"] == 0.0
    assert v["pinn_full"] == dict(range_weight=0.1, parity_weight=2.0, curvature_weight=3.0)


def test_summarize_averages_over_seeds_and_keeps_nan_reference_rows():
    rows = []
    for seed, mae in ((1, 0.01), (2, 0.03)):
        rows.append(dict(seed=seed, split="holdout", method="pinn_full", mae=mae, rmse=mae,
                         r2=0.9, parity_rms=0.001, curvature_rms=0.002))
        rows.append(dict(seed=seed, split="holdout", method="ground_truth", mae=np.nan,
                         rmse=np.nan, r2=np.nan, parity_rms=0.003, curvature_rms=0.004))
    summary = summarize(pd.DataFrame(rows)).set_index("method")
    assert summary.loc["pinn_full", "mae_mean"] == 0.02
    assert summary.loc["pinn_full", "n_seeds"] == 2
    assert np.isnan(summary.loc["ground_truth", "mae_mean"])
    assert summary.loc["ground_truth", "parity_rms_mean"] == 0.003


STRIKES = np.array([90.0, 100.0, 110.0, 90.0, 100.0, 120.0])


def test_merged_base_keeps_observed_and_fills_hidden_and_empty_rows():
    values = np.array([[0.30, 0.0, 0.26, 0.0, 0.22, 0.0], [0.2] * 6])
    observed = np.array([[True, False, True, False, True, False], [False] * 6])
    base = _merged_base(values, observed, STRIKES, fill_value=0.5)
    assert base[0, 0] == 0.30 and base[0, 2] == 0.26 and base[0, 4] == 0.22  # kept
    assert base[0, 1] == pytest.approx(0.22)   # call@100 <- pooled knot at 100 (put observed)
    assert base[0, 3] == pytest.approx(0.30)   # put@90 <- call@90
    assert (base[1] == 0.5).all()              # nothing observed -> fill value
    assert np.isfinite(base).all()


def test_untrained_residual_model_returns_the_base_exactly():
    rng = np.random.default_rng(0)
    values = 0.2 + 0.02 * rng.random((6, 6))
    observed = rng.random((6, 6)) < 0.5
    observed[:, 0] = True
    coords = np.zeros((6, 2))
    base = _merged_base(values, observed, STRIKES, 0.2)
    model = _fit_pinn(values, coords, observed, epochs=0, seed=0, residual_base=base)
    inputs = _residual_inputs(values, observed, coords, STRIKES, 0.2)
    out = model(torch.from_numpy(inputs)).detach().numpy()
    np.testing.assert_allclose(out, base, atol=1e-6)


def test_residual_model_trains_and_stays_finite():
    rng = np.random.default_rng(1)
    values = 0.2 + 0.02 * rng.random((10, 6))
    observed = rng.random((10, 6)) < 0.5
    observed[:, 0] = True
    coords = np.zeros((10, 2))
    base = _merged_base(values, observed, STRIKES, 0.2)
    model = _fit_pinn(values, coords, observed, epochs=3, seed=0, residual_base=base)
    out = model(torch.from_numpy(_residual_inputs(values, observed, coords, STRIKES, 0.2)))
    assert torch.isfinite(out).all() and out.shape == values.shape


def test_residual_variants_toggle_penalties():
    v = residual_variants(0.1, 2.0, 3.0)
    assert v["pinn_residual"]["parity_weight"] == 0.0
    assert v["pinn_residual_full"]["parity_weight"] == 2.0
    assert v["pinn_residual_full"]["curvature_weight"] == 3.0
