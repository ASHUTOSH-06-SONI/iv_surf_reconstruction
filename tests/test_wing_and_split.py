import numpy as np
import pandas as pd
import pytest

from src.evaluate.masked_quote_benchmark import (
    MODEL_NAMES,
    _evaluate_classical,
    _merged_interpolation,
    _merged_wing,
    _prepare_surfaces,
    _split_pools,
)

UNDERLYING = 100.0
STRIKES = np.array([80.0, 90.0, 100.0, 110.0, 120.0, 130.0] * 2)
IS_CALL = np.array([True] * 6 + [False] * 6)


def _setup(smile):
    unique, slot = np.unique(STRIKES, return_inverse=True)
    values = smile(np.log(STRIKES / UNDERLYING))
    return values, unique, slot


def test_quadratic_wing_is_exact_on_a_parabola_and_beats_flat():
    smile = lambda x: 0.2 + 0.3 * x + 2.0 * x**2
    values, unique, slot = _setup(smile)
    observed = (STRIKES >= 90) & (STRIKES <= 110)            # wings 80 and 120/130 hidden
    quad = _merged_wing(values, observed, STRIKES, unique, slot, UNDERLYING, "quadratic")
    flat = _merged_interpolation(values, observed, STRIKES, unique, slot)
    hidden = ~observed
    np.testing.assert_allclose(quad[hidden], values[hidden], atol=1e-9)
    assert np.abs(flat[hidden] - values[hidden]).max() > 0.01


def test_linear_wing_is_exact_on_a_line():
    smile = lambda x: 0.25 - 0.8 * x
    values, unique, slot = _setup(smile)
    observed = (STRIKES >= 90) & (STRIKES <= 110)
    out = _merged_wing(values, observed, STRIKES, unique, slot, UNDERLYING, "linear")
    np.testing.assert_allclose(out[~observed], values[~observed], atol=1e-9)


def test_wing_equals_merged_inside_the_observed_range():
    smile = lambda x: 0.2 + 2.0 * x**2
    values, unique, slot = _setup(smile)
    observed = (STRIKES >= 90) & (STRIKES <= 120) & (np.arange(12) % 2 == 0)
    merged = _merged_interpolation(values, observed, STRIKES, unique, slot)
    for mode in ("linear", "quadratic"):
        wing = _merged_wing(values, observed, STRIKES, unique, slot, UNDERLYING, mode)
        inside = (~observed) & (STRIKES >= STRIKES[observed].min()) & (STRIKES <= STRIKES[observed].max())
        np.testing.assert_allclose(wing[inside], merged[inside])


def test_wing_falls_back_to_flat_with_one_knot_and_clips_to_valid_iv():
    values, unique, slot = _setup(lambda x: 0.2 + 0 * x)
    one = np.zeros(12, bool)
    one[2] = True
    out = _merged_wing(values, one, STRIKES, unique, slot, UNDERLYING, "quadratic")
    assert np.allclose(out[~one & (STRIKES != 100.0)], 0.2)
    steep = np.array([0.9, 0.5, 0.2, 0.05, 0.02, 0.02] * 2)[: 12]
    observed = np.zeros(12, bool)
    observed[[1, 2, 3]] = True
    out = _merged_wing(steep, observed, STRIKES, unique, slot, UNDERLYING, "linear")
    assert np.nanmin(out[~observed]) >= 0.01 and np.nanmax(out[~observed]) <= 3.0
    with pytest.raises(ValueError):
        _merged_wing(values, one, STRIKES, unique, slot, UNDERLYING, "cubic")


def test_classical_dispatch_covers_wing_methods():
    assert {"merged_linear_wing", "merged_quad_wing"} <= set(MODEL_NAMES)
    values, _, _ = _setup(lambda x: 0.2 + 0.5 * x**2)
    observed = ((STRIKES >= 90) & (STRIKES <= 110))[None, :]
    frame = pd.DataFrame({"underlying": [UNDERLYING]})
    for method in ("merged_linear_wing", "merged_quad_wing"):
        pred, failures = _evaluate_classical(
            method, frame, values[None, :], observed, STRIKES, IS_CALL, ~observed
        )
        assert failures == 0 and np.isfinite(pred[~observed]).all()


def _frame():
    n = 300
    ts = 1_745_300_000_000_000_000 + np.arange(n) * 60_000_000_000
    expiry = np.array(["2025-04-24"] * 100 + ["2025-04-30"] * 100 + ["2025-05-08"] * 100)
    return pd.DataFrame({"timestamp": ts, "expiry": expiry})


def test_expiry_split_holds_out_the_latest_expiry_entirely():
    frame = _frame()
    train, val, hold = _split_pools(frame, seed=0, split_mode="expiry", holdout_expiry=None)
    expiries = frame["expiry"].to_numpy()
    assert set(expiries[train]) == {"2025-04-24", "2025-04-30"}
    assert set(expiries[val]) == set(expiries[hold]) == {"2025-05-08"}
    assert val.max() < hold.min() and len(val) == len(hold) == 50
    assert set(train).isdisjoint(val) and set(train).isdisjoint(hold)


def test_expiry_split_honours_explicit_expiry_and_rejects_bad_input():
    frame = _frame()
    train, val, hold = _split_pools(frame, 0, "expiry", "2025-04-30")
    assert set(frame["expiry"].to_numpy()[train]) == {"2025-04-24", "2025-05-08"}
    with pytest.raises(ValueError):
        _split_pools(frame, 0, "expiry", "2030-01-01")
    with pytest.raises(ValueError):
        _split_pools(frame, 0, "random", None)


def test_temporal_split_is_still_time_ordered():
    train, val, hold = _split_pools(_frame(), 0, "temporal", None)
    assert train.max() < val.min() and val.max() < hold.min()


def test_prepare_surfaces_expiry_mode_never_trains_on_the_held_out_expiry(tmp_path):
    rng = np.random.default_rng(0)
    frame = _frame()
    frame["underlying"] = 23000 + np.cumsum(rng.normal(0, 3, len(frame)))
    strikes = list(range(22800, 23400, 100))
    for side in ("call", "put"):
        for k in strikes:
            frame[f"{side}_iv_{k}"] = 0.15 + rng.normal(0, 0.002, len(frame))
    test = frame.drop(columns=["expiry"]).iloc[:50].copy()
    test["timestamp"] = np.arange(50)
    for c in test.columns:
        if c.startswith(("call_iv", "put_iv")):
            test.loc[rng.random(50) < 0.6, c] = np.nan
    frame.to_parquet(tmp_path / "train.parquet")
    test.to_parquet(tmp_path / "test.parquet")
    data = _prepare_surfaces(
        str(tmp_path / "train.parquet"), str(tmp_path / "test.parquet"),
        train_rows=100, evaluation_rows=20, seed=1, split_mode="expiry",
    )
    assert set(data.val_frame["expiry"].astype(str)) == {"2025-05-08"}
    assert set(data.test_frame["expiry"].astype(str)) == {"2025-05-08"}
    assert len(data.train_values) == 100
    assert "2025-05-08" not in set(data.train_frame["expiry"].astype(str))
