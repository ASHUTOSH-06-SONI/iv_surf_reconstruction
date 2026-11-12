import pandas as pd
import pytest

from src.data.reshape_deribit_surfaces import reshape_snapshots


def _snapshots():
    return pd.DataFrame(
        [
            {
                "snapshot_time_utc": "2026-11-12T12:00:00+00:00",
                "snapshot_timestamp_ms": 1_794_484_800_000,
                "quote_timestamp_ms": 1_794_484_790_000,
                "instrument_name": "ETH-13NOV26-3000-C",
                "expiration_timestamp": 1_794_556_800_000,
                "strike": 3000,
                "option_type": "call",
                "mark_price": 0.1,
                "mark_iv": 0.6,
            },
            {
                "snapshot_time_utc": "2026-11-12T12:00:00+00:00",
                "snapshot_timestamp_ms": 1_794_484_800_000,
                "quote_timestamp_ms": 1_794_484_700_000,
                "instrument_name": "ETH-13NOV26-3000-P",
                "expiration_timestamp": 1_794_556_800_000,
                "strike": 3000,
                "option_type": "put",
                "mark_price": 0.12,
                "mark_iv": 0.7,
            },
        ]
    )


def test_reshape_pairs_calls_and_puts_and_retains_quote_age():
    surfaces = reshape_snapshots(_snapshots())

    assert len(surfaces) == 1
    row = surfaces.iloc[0]
    assert row["call_iv"] == 0.6
    assert row["put_iv"] == 0.7
    assert row["call_quote_age_seconds"] == 10
    assert row["put_quote_age_seconds"] == 100
    assert row["strike"] == 3000
    assert row["surface_id"] == "1794484800000_1794556800000"


def test_reshape_preserves_one_sided_strikes_with_missing_counterpart():
    surfaces = reshape_snapshots(_snapshots().iloc[:1])

    assert len(surfaces) == 1
    assert surfaces.iloc[0]["call_iv"] == 0.6
    assert pd.isna(surfaces.iloc[0]["put_iv"])
    assert pd.isna(surfaces.iloc[0]["put_quote_age_seconds"])


def test_reshape_rejects_duplicate_option_rows():
    duplicated = pd.concat([_snapshots(), _snapshots().iloc[:1]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate option rows"):
        reshape_snapshots(duplicated)


def test_reshape_rejects_expiry_before_snapshot():
    snapshots = _snapshots()
    snapshots["expiration_timestamp"] = snapshots["snapshot_timestamp_ms"] - 1

    with pytest.raises(ValueError, match="expiry precedes the snapshot"):
        reshape_snapshots(snapshots)
