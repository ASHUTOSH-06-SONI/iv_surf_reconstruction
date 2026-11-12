"""Pair Deribit call/put mark IVs into expiry-by-strike surface rows."""

import argparse
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "snapshot_time_utc",
    "snapshot_timestamp_ms",
    "quote_timestamp_ms",
    "instrument_name",
    "expiration_timestamp",
    "strike",
    "option_type",
    "mark_price",
    "mark_iv",
}
SURFACE_COLUMNS = [
    "surface_id",
    "snapshot_time_utc",
    "snapshot_timestamp_ms",
    "expiration_time_utc",
    "expiration_timestamp",
    "time_to_expiry_seconds",
    "strike",
    "call_iv",
    "put_iv",
    "call_mark_price",
    "put_mark_price",
    "call_quote_timestamp_ms",
    "put_quote_timestamp_ms",
    "call_quote_age_seconds",
    "put_quote_age_seconds",
]


def reshape_snapshots(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Return one row per snapshot, expiry, and strike, retaining quote ages."""
    missing_columns = REQUIRED_COLUMNS - set(snapshots.columns)
    if missing_columns:
        raise ValueError(f"Snapshot data is missing columns: {sorted(missing_columns)}")
    if snapshots.empty:
        raise ValueError("Snapshot data contains no option rows.")

    data = snapshots.copy()
    for column in ("snapshot_timestamp_ms", "expiration_timestamp", "strike"):
        data[column] = pd.to_numeric(data[column], errors="raise")
    for column in ("quote_timestamp_ms", "mark_price", "mark_iv"):
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["option_type"] = data["option_type"].astype(str).str.lower()
    invalid_types = sorted(set(data["option_type"]) - {"call", "put"})
    if invalid_types:
        raise ValueError(f"Unexpected option types: {invalid_types}")

    key = ["snapshot_timestamp_ms", "expiration_timestamp", "strike", "option_type"]
    if data.duplicated(key).any():
        raise ValueError("Duplicate option rows found for a snapshot, expiry, and strike.")

    pivot = data.pivot(
        index=["snapshot_timestamp_ms", "expiration_timestamp", "strike"],
        columns="option_type",
        values=["mark_iv", "mark_price", "quote_timestamp_ms"],
    )
    pivot.columns = [f"{value}_{option_type}" for value, option_type in pivot.columns]
    pivot = pivot.reset_index()

    for column in (
        "mark_iv_call",
        "mark_iv_put",
        "mark_price_call",
        "mark_price_put",
        "quote_timestamp_ms_call",
        "quote_timestamp_ms_put",
    ):
        if column not in pivot:
            pivot[column] = pd.NA

    snapshot_times = data.groupby("snapshot_timestamp_ms")["snapshot_time_utc"].first()
    pivot["snapshot_time_utc"] = pivot["snapshot_timestamp_ms"].map(snapshot_times)
    pivot["expiration_time_utc"] = pd.to_datetime(
        pivot["expiration_timestamp"], unit="ms", utc=True
    ).dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    pivot["snapshot_datetime"] = pd.to_datetime(
        pivot["snapshot_timestamp_ms"], unit="ms", utc=True
    )
    pivot["expiration_datetime"] = pd.to_datetime(
        pivot["expiration_timestamp"], unit="ms", utc=True
    )
    pivot["time_to_expiry_seconds"] = (
        pivot["expiration_datetime"] - pivot["snapshot_datetime"]
    ).dt.total_seconds()
    if (pivot["time_to_expiry_seconds"] < 0).any():
        raise ValueError("Found option rows whose expiry precedes the snapshot.")
    pivot["surface_id"] = (
        pivot["snapshot_timestamp_ms"].astype("int64").astype(str)
        + "_"
        + pivot["expiration_timestamp"].astype("int64").astype(str)
    )
    for side in ("call", "put"):
        pivot[f"{side}_quote_age_seconds"] = (
            pivot["snapshot_timestamp_ms"] - pivot[f"quote_timestamp_ms_{side}"]
        ) / 1000

    result = pivot.rename(
        columns={
            "mark_iv_call": "call_iv",
            "mark_iv_put": "put_iv",
            "mark_price_call": "call_mark_price",
            "mark_price_put": "put_mark_price",
            "quote_timestamp_ms_call": "call_quote_timestamp_ms",
            "quote_timestamp_ms_put": "put_quote_timestamp_ms",
        }
    )
    result["snapshot_timestamp_ms"] = result["snapshot_timestamp_ms"].astype("int64")
    result["expiration_timestamp"] = result["expiration_timestamp"].astype("int64")
    return result[SURFACE_COLUMNS].sort_values(
        ["snapshot_timestamp_ms", "expiration_timestamp", "strike"]
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("data/raw/deribit/eth_option_snapshots.csv")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/deribit/eth_iv_surfaces.csv")
    )
    args = parser.parse_args()
    snapshots = pd.read_csv(args.input)
    surfaces = reshape_snapshots(snapshots)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    surfaces.to_csv(args.output, index=False)
    surface_count = surfaces["surface_id"].nunique()
    print(
        f"Saved {len(surfaces)} strike rows across {surface_count} "
        f"snapshot-expiry surfaces to {args.output}"
    )
    print("All quotes are retained; quote-age columns indicate update freshness.")


if __name__ == "__main__":
    main()
