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


class DeviceSession:
    def __init__(self, device_id: str, socket: WebSocket):
        self.device_id = device_id
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
    def __init__(self) -> None:
        self._sessions: dict[str, DeviceSession] = {}

    def add(self, session: DeviceSession) -> None:
        self._sessions[session.device_id] = session
        log.info("device connected: %s", session.device_id)

    def remove(self, device_id: str) -> None:
        self._sessions.pop(device_id, None)
        log.info("device disconnected: %s", device_id)

    def get(self, device_id: str) -> DeviceSession | None:
        return self._sessions.get(device_id)

    def any(self) -> DeviceSession | None:
        return next(iter(self._sessions.values()), None)
