"""Collect read-only Deribit ETH option mark-IV snapshots over WebSocket."""

import argparse
import asyncio
import csv
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, TextIO

LOGGER = logging.getLogger(__name__)
WEBSOCKET_URL = "wss://www.deribit.com/ws/api/v2"
CHANNEL = "markprice.options.eth_usd"
OUTPUT_COLUMNS = [
    "snapshot_time_utc",
    "snapshot_timestamp_ms",
    "quote_timestamp_ms",
    "instrument_name",
    "index_name",
    "expiration_timestamp",
    "strike",
    "option_type",
    "mark_price",
    "mark_iv",
    "instrument_count",
    "observed_instrument_count",
    "coverage",
]


def _instrument_map(instruments: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Keep option metadata needed to arrange streamed IVs into surfaces."""
    metadata = {}
    for instrument in instruments:
        name = instrument.get("instrument_name")
        if not name or instrument.get("kind") != "option":
            continue
        metadata[str(name)] = {
            "index_name": instrument.get("index_name") or "eth_usd",
            "expiration_timestamp": instrument.get("expiration_timestamp"),
            "strike": instrument.get("strike"),
            "option_type": instrument.get("option_type"),
        }
    if not metadata:
        raise ValueError("Deribit returned no ETH option instruments.")
    return metadata


def _apply_notification(
    message: Mapping[str, Any],
    instruments: Mapping[str, Mapping[str, Any]],
    state: Dict[str, Dict[str, Any]],
) -> None:
    """Update latest mark values from a Deribit subscription notification."""
    if message.get("method") != "subscription":
        return
    params = message.get("params")
    if not isinstance(params, Mapping) or params.get("channel") != CHANNEL:
        return
    data = params.get("data")
    if not isinstance(data, list):
        raise ValueError("Malformed Deribit mark-price notification: data must be a list.")
    for quote in data:
        if not isinstance(quote, Mapping):
            continue
        name = quote.get("instrument_name")
        if name not in instruments:
            continue
        state[str(name)] = {
            "quote_timestamp_ms": quote.get("timestamp"),
            "mark_price": quote.get("mark_price"),
            "mark_iv": quote.get("iv"),
        }


def _snapshot_rows(
    instruments: Mapping[str, Mapping[str, Any]],
    state: Mapping[str, Mapping[str, Any]],
    timestamp_ms: int,
) -> List[Dict[str, Any]]:
    """Create one wide-chain snapshot in tidy, one-row-per-option form."""
    snapshot_time = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat()
    observed_count = sum(
        name in state and state[name].get("mark_iv") is not None for name in instruments
    )
    coverage = observed_count / len(instruments)
    rows = []
    for name, metadata in instruments.items():
        quote = state.get(name, {})
        rows.append(
            {
                "snapshot_time_utc": snapshot_time,
                "snapshot_timestamp_ms": timestamp_ms,
                "quote_timestamp_ms": quote.get("quote_timestamp_ms"),
                "instrument_name": name,
                **metadata,
                "mark_price": quote.get("mark_price"),
                "mark_iv": quote.get("mark_iv"),
                "instrument_count": len(instruments),
                "observed_instrument_count": observed_count,
                "coverage": coverage,
            }
        )
    return rows


def _open_output(path: Path) -> TextIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    output = path.open("a+", newline="", encoding="utf-8")
    if exists:
        output.seek(0)
        existing_header = next(csv.reader([output.readline().rstrip("\r\n")]), [])
        output.seek(0, 2)
        if existing_header != OUTPUT_COLUMNS:
            output.close()
            raise ValueError(f"Existing output file has an incompatible header: {path}")
    else:
        csv.DictWriter(output, fieldnames=OUTPUT_COLUMNS).writeheader()
        output.flush()
    return output


async def _rpc_call(
    websocket: Any,
    request_id: int,
    method: str,
    params: Mapping[str, Any],
    on_message: Any,
) -> Mapping[str, Any]:
    request = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": dict(params),
    }
    await websocket.send(json.dumps(request))
    deadline = time.monotonic() + 15
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Timed out waiting for Deribit {method} response.")
        raw_message = await asyncio.wait_for(websocket.recv(), timeout=remaining)
        message = json.loads(raw_message)
        if message.get("id") == request_id:
            if "error" in message:
                raise RuntimeError(f"Deribit {method} failed: {message['error']}")
            return message
        on_message(message)


async def collect(
    output_path: Path,
    snapshot_interval: float,
    duration: float,
    websocket_url: str = WEBSOCKET_URL,
) -> None:
    try:
        import websockets
    except ImportError as exc:
        raise RuntimeError(
            "The WebSocket dependency is missing. Install project requirements first."
        ) from exc

    async with websockets.connect(
        websocket_url, ping_interval=20, ping_timeout=20, close_timeout=10
    ) as websocket:
        instruments_response = await _rpc_call(
            websocket,
            1,
            "public/get_instruments",
            {"currency": "ETH", "kind": "option", "expired": False},
            lambda _message: None,
        )
        result = instruments_response.get("result")
        if not isinstance(result, list):
            raise ValueError("Unexpected response from Deribit public/get_instruments.")
        instruments = _instrument_map(result)
        state: Dict[str, Dict[str, Any]] = {}

        def update_state(message: Mapping[str, Any]) -> None:
            _apply_notification(message, instruments, state)

        await _rpc_call(
            websocket,
            2,
            "public/subscribe",
            {"channels": [CHANNEL]},
            update_state,
        )
        LOGGER.info("Tracking %d ETH options on %s", len(instruments), CHANNEL)
        LOGGER.info("Writing snapshots to %s", output_path)

        with _open_output(output_path) as output:
            writer = csv.DictWriter(output, fieldnames=OUTPUT_COLUMNS)
            deadline = time.monotonic() + duration
            next_snapshot = time.monotonic() + snapshot_interval
            while time.monotonic() < deadline:
                wait_seconds = min(next_snapshot, deadline) - time.monotonic()
                if wait_seconds > 0:
                    try:
                        raw_message = await asyncio.wait_for(
                            websocket.recv(), timeout=wait_seconds
                        )
                    except asyncio.TimeoutError:
                        pass
                    else:
                        update_state(json.loads(raw_message))
                if time.monotonic() >= next_snapshot:
                    rows = _snapshot_rows(instruments, state, int(time.time() * 1000))
                    writer.writerows(rows)
                    output.flush()
                    LOGGER.info(
                        "Snapshot saved: %d/%d options have mark IV (%.1f%% coverage)",
                        rows[0]["observed_instrument_count"],
                        len(instruments),
                        rows[0]["coverage"] * 100,
                    )
                    next_snapshot += snapshot_interval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("data/raw/deribit/eth_option_snapshots.csv")
    )
    parser.add_argument("--snapshot-interval", type=float, default=60.0, help="Seconds")
    parser.add_argument("--duration", type=float, default=3600.0, help="Collection seconds")
    parser.add_argument("--websocket-url", default=WEBSOCKET_URL)
    args = parser.parse_args()
    if args.snapshot_interval <= 0 or args.duration <= 0:
        parser.error("--snapshot-interval and --duration must be positive.")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    asyncio.run(collect(args.output, args.snapshot_interval, args.duration, args.websocket_url))


if __name__ == "__main__":
    main()
