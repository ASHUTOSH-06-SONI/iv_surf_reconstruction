import csv

import pytest

from src.data.deribit_collector import (
    CHANNEL,
    OUTPUT_COLUMNS,
    _apply_notification,
    _instrument_map,
    _open_output,
    _snapshot_rows,
)


INSTRUMENTS = [
    {
        "instrument_name": "ETH-30DEC26-3000-C",
        "kind": "option",
        "index_name": "eth_usd",
        "expiration_timestamp": 1_798_603_200_000,
        "strike": 3000,
        "option_type": "call",
    },
    {
        "instrument_name": "ETH-30DEC26-3000-P",
        "kind": "option",
        "index_name": "eth_usd",
        "expiration_timestamp": 1_798_603_200_000,
        "strike": 3000,
        "option_type": "put",
    },
    {"instrument_name": "ETH-PERPETUAL", "kind": "future"},
]


def test_instrument_map_keeps_only_options_and_surface_coordinates():
    instruments = _instrument_map(INSTRUMENTS)

    assert set(instruments) == {
        "ETH-30DEC26-3000-C",
        "ETH-30DEC26-3000-P",
    }
    assert instruments["ETH-30DEC26-3000-C"]["option_type"] == "call"
    assert instruments["ETH-30DEC26-3000-P"]["strike"] == 3000
    assert instruments["ETH-30DEC26-3000-C"]["index_name"] == "eth_usd"


def test_instrument_map_rejects_empty_option_lists():
    with pytest.raises(ValueError, match="no ETH option instruments"):
        _instrument_map([{"instrument_name": "ETH-PERPETUAL", "kind": "future"}])


def test_subscription_updates_are_mapped_and_unknown_instruments_ignored():
    instruments = _instrument_map(INSTRUMENTS)
    state = {}
    _apply_notification(
        {
            "method": "subscription",
            "params": {
                "channel": CHANNEL,
                "data": [
                    {
                        "instrument_name": "ETH-30DEC26-3000-C",
                        "timestamp": 1_798_560_000_000,
                        "mark_price": 0.1,
                        "iv": 0.65,
                    },
                    {"instrument_name": "ETH-UNKNOWN", "iv": 0.4},
                ],
            },
        },
        instruments,
        state,
    )

    assert state == {
        "ETH-30DEC26-3000-C": {
            "quote_timestamp_ms": 1_798_560_000_000,
            "mark_price": 0.1,
            "mark_iv": 0.65,
        }
    }


def test_snapshot_rows_include_missing_options_and_report_coverage():
    instruments = _instrument_map(INSTRUMENTS)
    state = {
        "ETH-30DEC26-3000-C": {
            "quote_timestamp_ms": 1_798_560_000_000,
            "mark_price": 0.1,
            "mark_iv": 0.65,
        }
    }
    rows = _snapshot_rows(instruments, state, 1_798_560_060_000)

    assert len(rows) == 2
    assert rows[0]["coverage"] == 0.5
    assert rows[0]["mark_iv"] == 0.65
    assert rows[1]["mark_iv"] is None
    assert rows[1]["instrument_count"] == 2


def test_output_file_writes_header_and_rejects_incompatible_existing_header(tmp_path):
    output_path = tmp_path / "snapshots.csv"
    with _open_output(output_path) as output:
        csv.DictWriter(output, fieldnames=OUTPUT_COLUMNS).writerow(
            {column: "" for column in OUTPUT_COLUMNS}
        )
    with _open_output(output_path):
        pass

    incompatible = tmp_path / "incompatible.csv"
    incompatible.write_text("wrong,header\n", encoding="utf-8")
    with pytest.raises(ValueError, match="incompatible header"):
        _open_output(incompatible)
