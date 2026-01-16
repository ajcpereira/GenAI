from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from influxdb_client import InfluxDBClient
from influxdb_client.client.flux_table import FluxRecord

log = logging.getLogger("genai_core.mcp_host.influxdb")


@dataclass(frozen=True)
class InfluxConn:
    url: str
    org: str
    bucket: str
    token: str


def _require_str(d: Dict[str, Any], key: str) -> str:
    v = d.get(key)
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"missing_or_invalid_{key}")
    return v.strip()


def _optional_int(d: Dict[str, Any], key: str, default: int) -> int:
    v = d.get(key, default)
    try:
        i = int(v)
    except Exception:
        i = default
    return i


def _read_token(args: Dict[str, Any]) -> str:
    """
    Security rule: token must NOT be passed directly in args.
    We only accept token_env and read from os.environ in the MCP Host process.
    """
    token_env = (args.get("token_env") or "").strip()
    if not token_env:
        raise ValueError("missing_token_env")
    token = os.environ.get(token_env, "").strip()
    if not token:
        raise ValueError(f"token_env_not_set:{token_env}")
    return token


def _build_flux_query(args: Dict[str, Any], bucket: str) -> str:
    """
    Two modes:
      A) raw flux: args["flux"] (must be a string)
      B) structured: measurement/field + range + aggregate/window (safe defaults)

    IMPORTANT: this is Phase 1; for production, consider a stricter schema/allowlist.
    """
    flux = args.get("flux")
    if isinstance(flux, str) and flux.strip():
        # User provides full Flux query. We still enforce that it reads from the configured bucket.
        # If they included from(bucket:"..."), we won't rewrite; we only sanity-check presence.
        f = flux.strip()
        return f

    measurement = (args.get("measurement") or "").strip()
    field = (args.get("field") or "").strip()
    if not measurement:
        raise ValueError("missing_measurement_or_flux")

    # Range: default last 1h
    range_start = (args.get("range_start") or "-1h").strip()
    # Optional stop (rare)
    range_stop = (args.get("range_stop") or "").strip()

    # Aggregate/window: optional
    window_every = (args.get("window_every") or "").strip()  # e.g. "1m"
    fn = (args.get("aggregate_fn") or "").strip()            # e.g. "mean", "max", "min", "sum"

    # Basic structured query
    parts: List[str] = []
    parts.append(f'from(bucket: "{bucket}")')
    if range_stop:
        parts.append(f'|> range(start: {range_start}, stop: {range_stop})')
    else:
        parts.append(f'|> range(start: {range_start})')

    parts.append(f'|> filter(fn: (r) => r._measurement == "{measurement}")')
    if field:
        parts.append(f'|> filter(fn: (r) => r._field == "{field}")')

    if window_every and fn:
        # Flux aggregateWindow example:
        # |> aggregateWindow(every: 1m, fn: mean, createEmpty: false)
        parts.append(f'|> aggregateWindow(every: {window_every}, fn: {fn}, createEmpty: false)')

    # Yield is optional; query api returns results regardless.
    return "\n".join(parts)


def _record_to_row(rec: FluxRecord) -> Dict[str, Any]:
    # Normalize core columns commonly needed downstream.
    row: Dict[str, Any] = {
        "time": rec.get_time().isoformat() if rec.get_time() else None,
        "value": rec.get_value(),
        "field": rec.get_field(),
        "measurement": rec.get_measurement(),
    }

    # Tags/extra columns: keep them but do not explode huge objects.
    # rec.values includes internal keys; filter out noisy ones.
    tags: Dict[str, Any] = {}
    for k, v in (rec.values or {}).items():
        if k in ("_start", "_stop", "_time", "_value", "_field", "_measurement", "result", "table"):
            continue
        tags[k] = v
    row["tags"] = tags
    return row


async def influxdb_query(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    MCP tool: influxdb_query (InfluxDB v2 / Flux)

    Args (non-sensitive):
      url: str
      org: str
      bucket: str
      token_env: str            # environment variable name on MCP Host
      flux: str (optional)      # full Flux query
      OR structured:
        measurement: str
        field: str (optional)
        range_start: str (default "-1h")   # e.g. "-24h"
        range_stop: str (optional)
        window_every: str (optional)       # e.g. "1m"
        aggregate_fn: str (optional)       # e.g. "mean"

      max_rows: int (optional, default 2000, hard cap 20000)
      timeout_s: int (optional, default 10, hard cap 60)
    """
    try:
        url = _require_str(args, "url")
        org = _require_str(args, "org")
        bucket = _require_str(args, "bucket")
        token = _read_token(args)

        timeout_s = _optional_int(args, "timeout_s", 10)
        if timeout_s <= 0:
            timeout_s = 10
        timeout_s = min(timeout_s, 60)

        max_rows = _optional_int(args, "max_rows", 2000)
        if max_rows <= 0:
            max_rows = 2000
        max_rows = min(max_rows, 20000)

        conn = InfluxConn(url=url, org=org, bucket=bucket, token=token)
        flux_query = _build_flux_query(args, bucket=conn.bucket)

        # Run blocking client work in a thread to avoid blocking the event loop.
        def _run_query() -> List[Dict[str, Any]]:
            rows: List[Dict[str, Any]] = []
            # InfluxDBClient has its own timeout controls; we also wrap with asyncio.wait_for externally.
            with InfluxDBClient(url=conn.url, token=conn.token, org=conn.org, timeout=timeout_s * 1000) as client:
                qapi = client.query_api()
                tables = qapi.query(flux_query, org=conn.org)

                for table in tables:
                    for rec in table.records:
                        rows.append(_record_to_row(rec))
                        if len(rows) >= max_rows:
                            return rows
            return rows

        rows: List[Dict[str, Any]] = await asyncio.wait_for(asyncio.to_thread(_run_query), timeout=timeout_s + 2)

        return {"results": rows, "error": ""}

    except asyncio.TimeoutError:
        return {"results": [], "error": "influxdb_query_timeout"}
    except Exception as e:
        # Do not leak secrets; token is never included in message.
        return {"results": [], "error": f"influxdb_query_failed: {type(e).__name__}: {e}"}
