"""
Memory manager — stores user-approved preferences and task context.

Explicitly NEVER stores secrets (PINs, OTPs, CVVs, passwords). In this reference
build it is in-process; swap the dict for PostgreSQL in production.
"""
from __future__ import annotations

import threading

_BANNED_KEYS = ("pin", "otp", "cvv", "password", "card", "secret")


class MemoryManager:
    def __init__(self) -> None:
        self._prefs: dict[str, dict[str, str]] = {}   # user_id -> {key: value}
        self._lock = threading.Lock()

    def remember(self, user_id: str, key: str, value: str) -> bool:
        if any(b in key.lower() for b in _BANNED_KEYS):
            return False  # refuse to store secrets
        with self._lock:
            self._prefs.setdefault(user_id, {})[key] = value
        return True

    def recall(self, user_id: str) -> dict[str, str]:
        with self._lock:
            return dict(self._prefs.get(user_id, {}))

    def forget(self, user_id: str, key: str | None = None) -> None:
        with self._lock:
            if key is None:
                self._prefs.pop(user_id, None)
            else:
                self._prefs.get(user_id, {}).pop(key, None)
