"""Utility module for fine‑grained performance timing & storage.
   Imported by **main.py** so business logic stays clean.
   ------------------------------------------------------------------
   Database DDL (run once):

   CREATE TABLE IF NOT EXISTS performance_metrics (
       id              SERIAL PRIMARY KEY,
       prompt          TEXT,
       request_start   TIMESTAMPTZ,
       request_end     TIMESTAMPTZ,
       duration_ms     INTEGER,
       iterations      INTEGER,
       success         BOOLEAN,
       error_message   TEXT,
       function_times  JSONB            -- {"func": ms, ...}
   );
"""
from __future__ import annotations

import json
import os
import time
from contextlib import closing
from typing import Callable, Dict, TypeVar

import psycopg2

__all__ = [
    "time_call",
    "record_metric",
]

T = TypeVar("T")

# Pick DSN from env or default.
DB_DSN = os.getenv("PG_DSN", "dbname=mydb user=myuser password=mypw host=postgres")

# ── Timing helper ──────────────────────────────────────────────

def time_call(fn_times: Dict[str, int], label: str, func: Callable[..., T], *args, **kwargs) -> T:  # type: ignore[name‑defined]
    """Run *func* and stash runtime in **fn_times[label]** (milliseconds)."""
    start = time.perf_counter()
    out: T = func(*args, **kwargs)
    fn_times[label] = int((time.perf_counter() - start) * 1000)
    return out

# ── Persistence helper ─────────────────────────────────────────

def record_metric(
    *,
    prompt: str,
    start_ts: float,
    iterations: int,
    success: bool,
    function_times: Dict[str, int],
    error_msg: str = "",
) -> None:
    """Insert one row into **performance_metrics**.  Errors are logged but not raised."""
    try:
        duration_ms = int((time.perf_counter() - start_ts) * 1000)
        with closing(psycopg2.connect(DB_DSN)) as conn, conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO performance_metrics
                      (prompt, request_start, request_end, duration_ms,
                       iterations, success, error_message, function_times)
                VALUES (%s, to_timestamp(%s), to_timestamp(%s), %s,
                        %s, %s, %s, %s::jsonb)
                """,
                (
                    prompt,
                    start_ts,
                    start_ts + duration_ms / 1000.0,
                    duration_ms,
                    iterations,
                    success,
                    error_msg[:500],
                    json.dumps(function_times),
                ),
            )
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️  metrics write failed: {exc}")