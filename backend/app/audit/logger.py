"""
Audit logger — append-only record of what the agent attempted, what the user
approved, and what was verified. Writes JSON lines to disk; never records secrets.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

log = logging.getLogger("agent.audit")

_AUDIT_DIR = Path(__file__).resolve().parents[2] / "audit_logs"
_REDACT = ("otp", "pin", "cvv", "password", "card")


def _redact(data: dict) -> dict:
    out = {}
    for k, v in data.items():
        if any(b in str(k).lower() for b in _REDACT):
            out[k] = "***redacted***"
        else:
            out[k] = v
    return out


# Optional durable sink (app.db.postgres.Database), set at startup when Postgres is up.
_db = None


def set_audit_db(db) -> None:
    global _db
    _db = db


def audit(task_id: str, kind: str, detail: str = "", **data) -> None:
    _AUDIT_DIR.mkdir(exist_ok=True)
    clean = _redact(data)
    record = {"ts": time.time(), "task_id": task_id, "kind": kind,
              "detail": detail, "data": clean}
    try:
        with (_AUDIT_DIR / f"{task_id}.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except OSError as e:
        log.warning("audit write failed: %s", e)
    # Append-only mirror in Postgres (blueprint: audit_events). Non-blocking, batched.
    if _db is not None:
        _db.enqueue_audit(task_id, "agent", kind, detail, clean)
