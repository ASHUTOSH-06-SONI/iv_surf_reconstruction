import numpy as np
import pandas as pd
import pytest

from src.evaluate.masked_quote_benchmark import (
    MODEL_NAMES,
    _counterpart_observed_fraction,
    _evaluate_classical,
    _merged_interpolation,
    _parity_copy,
    _partner_index,
    _region_masks,
    parity_gap_report,
)

# columns: calls at 90/100/110, puts at 90/100/120
STRIKES = np.array([90.0, 100.0, 110.0, 90.0, 100.0, 120.0])
IS_CALL = np.array([True, True, True, False, False, False])


def test_partner_index_pairs_same_strike_opposite_side():
    assert _partner_index(STRIKES, IS_CALL).tolist() == [3, 4, -1, 0, 1, -1]


def test_parity_copy_uses_counterpart_else_side_interpolation():
    partner = _partner_index(STRIKES, IS_CALL)
    values = np.array([0.30, 0.20, 0.25, 0.31, 0.21, 0.40])
    observed = np.array([True, False, True, False, True, True])
    out = _parity_copy(values, observed, STRIKES, IS_CALL, partner)
    assert out[3] == pytest.approx(0.30)   # put@90 <- observed call@90
    assert out[1] == pytest.approx(0.21)   # call@100 <- observed put@100 (not interpolated 0.275)
    assert np.isnan(out[0]) and np.isnan(out[2])  # observed quotes are left unpredicted


def test_parity_copy_falls_back_to_interpolation_without_counterpart():
    partner = _partner_index(STRIKES, IS_CALL)
    values = np.array([0.30, 0.20, 0.26, 0.31, 0.22, 0.40])
    observed = np.array([True, False, True, False, False, True])  # no counterpart for call@100
    out = _parity_copy(values, observed, STRIKES, IS_CALL, partner)
    assert out[1] == pytest.approx(0.28)   # linear between 0.30@90 and 0.26@110
    assert out[3] == pytest.approx(0.30)   # still copied from call@90


def test_merged_interpolation_pools_both_sides_and_is_exact_at_knots():
    unique, slot = np.unique(STRIKES, return_inverse=True)
    values = np.array([0.30, 0.20, 0.25, 0.32, 0.21, 0.40])
    observed = np.array([True, False, False, True, True, False])
    out = _merged_interpolation(values, observed, STRIKES, unique, slot)
    # pooled: 90 -> mean(0.30, 0.32)=0.31 ; 100 -> 0.21 (put only)
    assert out[1] == pytest.approx(0.21)               # call@100 reads the pooled knot
    assert out[2] == pytest.approx(0.21)               # 110: flat extrapolation beyond last knot
    assert out[5] == pytest.approx(0.21)
    assert np.isnan(out[0]) and np.isnan(out[3])


def test_merged_interpolation_uses_info_from_the_other_side():
    unique, slot = np.unique(STRIKES, return_inverse=True)
    values = np.array([0.30, 0.0, 0.0, 0.0, 0.0, 0.50])
    observed = np.array([True, False, False, False, False, True])  # call@90, put@120 only
    out = _merged_interpolation(values, observed, STRIKES, unique, slot)
    assert out[1] == pytest.approx(0.30 + (0.50 - 0.30) * 10 / 30)  # call@100 uses put@120
    assert out[2] == pytest.approx(0.30 + (0.50 - 0.30) * 20 / 30)


def test_merged_interpolation_returns_nan_without_observations():
    unique, slot = np.unique(STRIKES, return_inverse=True)
    out = _merged_interpolation(np.full(6, 0.2), np.zeros(6, bool), STRIKES, unique, slot)
    assert np.isnan(out).all()


def test_classical_dispatch_covers_new_methods():
    assert "parity_copy" in MODEL_NAMES and "merged_interpolation" in MODEL_NAMES
    values = np.array([[0.30, 0.20, 0.25, 0.31, 0.21, 0.40]])
    observed = np.array([[True, False, True, False, True, True]])
    hidden = ~observed
    for method in ("parity_copy", "merged_interpolation"):
        pred, failures = _evaluate_classical(
            method, pd.DataFrame(index=[0]), values, observed, STRIKES, IS_CALL, hidden
        )
        assert failures == 0 and np.isfinite(pred[hidden]).all()


def test_counterpart_fraction_and_gap_report():
    partner = _partner_index(STRIKES, IS_CALL)
    observed = np.array([[True, False, False, False, True, False]])
    hidden = ~observed
    # hidden: call@100 (partner put@100 observed), call@110 (none), put@90 (partner call@90 observed),
    #         put@100? observed -> excluded, put@120 (none)
    assert _counterpart_observed_fraction(observed, hidden, partner) == pytest.approx(2 / 4)

    cols = ["call_iv_90", "call_iv_100", "call_iv_110", "put_iv_90", "put_iv_100", "put_iv_120"]
    frame = pd.DataFrame([[0.30, 0.20, 0.25, 0.31, 0.24, 0.40]], columns=cols)
    report = parity_gap_report(frame, cols, STRIKES, IS_CALL)
    assert report["n_pairs"] == 2
    assert report["max_abs"] == pytest.approx(0.04)
    assert report["median_abs"] == pytest.approx(0.025)


def test_region_masks_split_by_observed_strike_range():
    values = np.full((1, 6), 0.2)
    observed = np.array([[False, True, False, False, True, False]])  # call@100, put@100 only
    hidden = ~observed
    regions = _region_masks(values, observed, hidden, STRIKES)
    # observed strike range is [100, 100]: only strike 100 quotes are interior (both observed here)
    assert not regions["interior"].any()
    assert regions["extrapolated"].sum() == 4

    observed = np.array([[True, False, False, False, False, True]])  # call@90, put@120
    regions = _region_masks(values, observed, ~observed, STRIKES)
    assert regions["interior"][0].tolist() == [False, True, True, True, True, False]
    assert not regions["extrapolated"].any()
    assert not (regions["interior"] & observed).any()


def test_region_masks_without_observations_are_all_extrapolated():
    values = np.full((1, 6), 0.2)
    observed = np.zeros((1, 6), dtype=bool)
    regions = _region_masks(values, observed, ~observed, STRIKES)
    assert regions["extrapolated"].all() and not regions["interior"].any()
