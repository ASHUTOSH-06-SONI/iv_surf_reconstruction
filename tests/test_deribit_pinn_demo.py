import numpy as np
import pandas as pd
import pytest

from src.evaluate.deribit_pinn_demo import (
    _chronological_split,
    _coordinates,
    _interpolate_row,
    _sample_observation_masks,
    _surface_arrays,
)


def _surface_frame():
    rows = []
    for snapshot in range(4):
        for strike, call_iv, put_iv in (
            (90.0, 0.30 + snapshot * 0.01, 0.35 + snapshot * 0.01),
            (100.0, 0.25 + snapshot * 0.01, 0.28 + snapshot * 0.01),
            (110.0, 0.27 + snapshot * 0.01, 0.32 + snapshot * 0.01),
        ):
            rows.append(
                {
                    "surface_id": f"{snapshot}_expiry",
                    "snapshot_time_utc": f"2026-01-01T00:0{snapshot}:00+00:00",
                    "snapshot_timestamp_ms": snapshot,
                    "expiration_time_utc": "2026-02-01T00:00:00+00:00",
                    "expiration_timestamp": 100,
                    "time_to_expiry_seconds": 100 - snapshot,
                    "strike": strike,
                    "call_iv": call_iv,
                    "put_iv": put_iv,
                    "call_quote_age_seconds": 1.0,
                    "put_quote_age_seconds": 2.0,
                }
            )
    return pd.DataFrame(rows)


def test_surface_arrays_builds_call_put_grid_and_ages():
    metadata, strikes, values, ages = _surface_arrays(_surface_frame())

    assert len(metadata) == 4
    assert strikes.tolist() == [90.0, 100.0, 110.0]
    assert values.shape == ages.shape == (4, 6)
    assert values[0].tolist() == [0.30, 0.25, 0.27, 0.35, 0.28, 0.32]
    assert ages[0].tolist() == [1, 1, 1, 2, 2, 2]


def test_mask_sampler_keeps_call_and_put_inputs_and_hides_finite_quotes():
    values = np.array([[0.3, 0.2, 0.4, 0.5, 0.6, 0.7]])

    observed, hidden = _sample_observation_masks(values, mask_fraction=0.5, seed=3)

    assert observed[:, :3].sum() >= 1
    assert observed[:, 3:].sum() >= 1
    assert np.all(observed | hidden)


def test_interpolation_baselines_fill_hidden_values():
    strikes = np.array([90.0, 100.0, 110.0])
    values = np.array([0.3, 0.0, 0.4, 0.5, 0.0, 0.7])
    observed = np.array([True, False, True, True, False, True])

    independent = _interpolate_row(values, observed, strikes, pooled=False)
    pooled = _interpolate_row(values, observed, strikes, pooled=True)

    assert independent[1] == pytest.approx(0.35)
    assert independent[4] == pytest.approx(0.6)
    assert pooled[1] == pytest.approx(0.475)
    assert pooled[4] == pytest.approx(0.475)


def test_chronological_split_keeps_timestamps_disjoint():
    metadata, _, _, _ = _surface_arrays(_surface_frame())

    train, validation, holdout = _chronological_split(metadata)

    sets = [
        set(metadata.iloc[index]["snapshot_timestamp_ms"])
        for index in (train, validation, holdout)
    ]
    assert sets == [{0, 1}, {2}, {3}]


def test_coordinates_encode_log_strike_and_maturity_per_quote():
    metadata, strikes, values, _ = _surface_arrays(_surface_frame())

    coordinates = _coordinates(strikes, values, metadata)

    assert coordinates.shape == (4, 2 * values.shape[1])
    np.testing.assert_allclose(coordinates[0, 0::2], np.tile(np.log(strikes / 100), 2))
    assert np.all(coordinates[:, 1::2] > 0)
