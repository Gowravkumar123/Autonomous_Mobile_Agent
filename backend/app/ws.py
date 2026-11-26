"""
Device transport. One WebSocket per paired device. The brain drives the device with
request/response calls (observe, act, handoff); responses are correlated by `rid`.

The same socket also carries fire-and-forget status/approval pushes to the companion
app UI.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import WebSocket

log = logging.getLogger("agent.ws")


def kind_of(device_id: str) -> str:
    """Executor kind from the device id: desktop companions pair as `desktop-<host>`."""
    return "desktop" if device_id.startswith("desktop-") else "phone"


class DeviceSession:
    def __init__(self, device_id: str, socket: WebSocket):
        self.device_id = device_id
        self.kind = kind_of(device_id)
        self.socket = socket
        self._rid = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._send_lock = asyncio.Lock()

    async def _send(self, msg: dict[str, Any]) -> None:
        async with self._send_lock:
            try:
                await self.socket.send_json(msg)
            except Exception as e:  # socket closed mid-send -> normalise to RuntimeError
                raise RuntimeError(f"device send failed: {type(e).__name__}") from e

    async def request(self, type_: str, timeout: float = 60.0, **data: Any) -> dict:
        """Send a request and await the device's matching response payload."""
        self._rid += 1
        rid = self._rid
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[rid] = fut
        await self._send({"type": type_, "rid": rid, **data})
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        finally:
            self._pending.pop(rid, None)

    async def push(self, type_: str, **data: Any) -> None:
        """Fire-and-forget UI update (status, approval_request, summary)."""
        await self._send({"type": type_, **data})

    def on_message(self, msg: dict[str, Any]) -> None:
        """Called by the ws endpoint for every inbound frame."""
        rid = msg.get("rid")
        if rid is not None and rid in self._pending:
            fut = self._pending[rid]
            if not fut.done():
                fut.set_result(msg.get("payload", {}))

    def fail_all(self, exc: Exception) -> None:
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(exc)
        self._pending.clear()


class DeviceRegistry:
    """Sessions on THIS worker (`_sessions`), plus proxies for devices connected to other
    workers (`_remote`, maintained by app.cluster). `get()` prefers the local socket."""

    def __init__(self) -> None:
        self._sessions: dict[str, DeviceSession] = {}
        self._remote: dict[str, Any] = {}
        self.cluster = None

    def add(self, session: DeviceSession) -> None:
        self._sessions[session.device_id] = session
        log.info("device connected: %s", session.device_id)

    def remove(self, device_id: str, session: DeviceSession | None = None) -> None:
        """Drop a device's session — but only if it is still the CURRENT one.

        After a flap the phone reconnects and its new session replaces the old one; the
        server notices the old socket died a little later. Removing by id alone would
        then delete the NEW, healthy session, leaving the phone connected but invisible
        to the backend (tasks failed with "device disconnected and did not return").
        """
        current = self._sessions.get(device_id)
        if session is not None and current is not session:
            log.info("stale session for %s closed (newer one kept)", device_id)
            return
        self._sessions.pop(device_id, None)
        log.info("device disconnected: %s", device_id)

    def get(self, device_id: str) -> DeviceSession | None:
        return self._sessions.get(device_id) or self._remote.get(device_id)

    def get_local(self, device_id: str) -> DeviceSession | None:
        return self._sessions.get(device_id)

    def all(self) -> list:
        """Every reachable session: local first, then remote proxies (cluster)."""
        return list(self._sessions.values()) + [r for k, r in self._remote.items()
                                                if k not in self._sessions]

    def any(self) -> DeviceSession | None:
        return next(iter(self.all()), None)
