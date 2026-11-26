"""
WebSocket device agent: the desktop's side of the protocol the phone speaks.

    server -> {"type": "observe", "rid": n}             reply {"rid": n, "payload": {"screen": ..}}
    server -> {"type": "act", "rid": n, "action": {..}} reply {"rid": n, "payload": {"ok": ..}}
    server -> {"type": "handoff", "rid": n, ...}        reply once the user has finished
    server -> {"type": "approval_request" | "ask_user" | "status" | ...}   (pushes)
    desktop -> {"type": "approval_response", task_id, approved} / {"type": "answer", ...}

Requests are handled concurrently (a handoff waiting for the user must not block an
observe), but all UI Automation work is serialized on the adapter's own thread.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from .safety import Safety
from .uia import WindowsDesktop

log = logging.getLogger("onepiece.desktop")


class DesktopAgent:
    def __init__(self, server: str, device_id: str, token: str, desktop: WindowsDesktop,
                 safety: Safety, *, handoff_timeout_s: float = 300.0,
                 resume_file: Path | None = None,
                 on_push: Callable[[dict], Awaitable[None] | None] | None = None):
        self.server = server.rstrip("/")
        self.device_id = device_id
        self.token = token
        self.desktop = desktop
        self.safety = safety
        self.handoff_timeout_s = handoff_timeout_s
        self.resume_file = resume_file or (safety.stop_file.parent / "RESUME")
        self.on_push = on_push
        self.ws = None
        self.closed = False
        self.auth_rejected = False
        self.connected = asyncio.Event()
        self._handoff_until = 0.0         # while the user is doing a step, refuse actions
        self._resume = asyncio.Event()
        self._tasks: set[asyncio.Task] = set()

    # ---- connection --------------------------------------------------------
    def ws_url(self) -> str:
        base = self.server.replace("https://", "wss://").replace("http://", "ws://")
        return f"{base}/ws/device/{self.device_id}"

    async def run_forever(self) -> None:
        import websockets
        delay = 1.0
        while not self.closed:
            try:
                async with websockets.connect(
                        self.ws_url(), additional_headers={"Authorization": f"Bearer {self.token}"},
                        max_size=8 * 1024 * 1024, ping_interval=20) as ws:
                    self.ws, delay = ws, 1.0
                    self.connected.set()
                    log.info("connected as %s", self.device_id)
                    async for raw in ws:
                        self._spawn(self._dispatch(json.loads(raw)))
            except asyncio.CancelledError:
                raise
            except Exception as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status in (401, 403):
                    self.auth_rejected = True
                    log.error("server rejected the device token (%s) — pair again", status)
                    return
                log.warning("connection lost (%s); retrying in %.0fs", type(e).__name__, delay)
            finally:
                self.ws = None
                self.connected.clear()
            if self.closed:
                break
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30.0)

    async def close(self) -> None:
        self.closed = True
        if self.ws is not None:
            await self.ws.close()
        for t in list(self._tasks):
            t.cancel()

    def _spawn(self, coro) -> None:
        t = asyncio.create_task(coro)
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)

    async def send(self, msg: dict) -> None:
        if self.ws is not None:
            await self.ws.send(json.dumps(msg))

    # ---- dispatch ----------------------------------------------------------
    async def _dispatch(self, msg: dict) -> None:
        rid = msg.get("rid")
        t = msg.get("type")
        try:
            if t == "observe":
                payload: dict[str, Any] = {"screen": await self.desktop.run(self.desktop.observe)}
            elif t == "act":
                payload = await self._act(msg.get("action") or {})
            elif t == "handoff":
                payload = await self._handoff(msg.get("approval") or {})
            else:
                await self._push(msg)
                return
        except Exception as e:
            log.exception("handler %s failed", t)
            payload = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        if rid is not None:
            await self.send({"rid": rid, "payload": payload})

    async def _act(self, action: dict) -> dict:
        kind = action.get("type")
        entry = {"event": "act", "type": kind, "ref": action.get("ref"),
                 "thought": (action.get("thought") or "")[:200]}
        reason = self.safety.check_kill()
        if reason:
            res = {"ok": False, "halt": True, "error": f"kill switch engaged: {reason}"}
        elif time.monotonic() < self._handoff_until:
            res = {"ok": False, "error": "the user is completing a step — waiting"}
        else:
            res = await self.desktop.run(self.desktop.act, action)
        meta = self.desktop._meta.get(action.get("ref"), {}) if action.get("ref") is not None else {}
        entry["target"] = meta.get("label", "")
        if kind == "type":
            entry["text"] = "***" if meta.get("password") else (action.get("text") or "")[:200]
        self.safety.log({**entry, "result": res})
        return res

    async def _handoff(self, approval: dict) -> dict:
        """A payment/sign-in step only the user may do. Stop acting, tell the user, and
        wait for them to resume (RESUME file, console Enter, or `resume()`)."""
        self._handoff_until = time.monotonic() + self.handoff_timeout_s
        self._resume.clear()
        self.safety.log({"event": "handoff", "reason": approval.get("reason", ""),
                         "summary": approval.get("summary", "")})
        await self._push({"type": "handoff", **approval,
                          "message": "Your turn: finish this step yourself, then resume "
                                     f"(create {self.resume_file} or press Enter)."})
        try:
            deadline = time.monotonic() + self.handoff_timeout_s
            while time.monotonic() < deadline:
                if self._resume.is_set() or self.resume_file.exists():
                    break
                await asyncio.sleep(0.25)
            else:
                return {"ok": True, "stop": True,
                        "message": "Handoff timed out — finish the step on your desktop."}
        finally:
            self._handoff_until = 0.0
            try:
                self.resume_file.unlink()
            except FileNotFoundError:
                pass
        self.safety.log({"event": "handoff_resumed"})
        return {"ok": True}

    def resume(self) -> None:
        self._resume.set()

    async def _push(self, msg: dict) -> None:
        if self.on_push is None:
            log.info("push: %s", {k: msg.get(k) for k in ("type", "title", "summary", "questions")})
            return
        r = self.on_push(msg)
        if asyncio.iscoroutine(r):
            await r

    # ---- user replies (console UI / tests) ---------------------------------
    async def approve(self, task_id: str, approved: bool, note: str = "") -> None:
        await self.send({"type": "approval_response", "task_id": task_id,
                         "approved": approved, "note": note})

    async def answer(self, task_id: str, text: str) -> None:
        await self.send({"type": "answer", "task_id": task_id,
                         "facts": {"clarification_answer": text}})
