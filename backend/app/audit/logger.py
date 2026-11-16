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


def audit(task_id: str, kind: str, detail: str = "", **data) -> None:
    _AUDIT_DIR.mkdir(exist_ok=True)
    record = {"ts": time.time(), "task_id": task_id, "kind": kind,
              "detail": detail, "data": _redact(data)}
    try:
        with (_AUDIT_DIR / f"{task_id}.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as e:
        log.warning("audit write failed: %s", e)
