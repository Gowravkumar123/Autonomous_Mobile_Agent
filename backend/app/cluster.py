"""
Multi-worker routing (blueprint: horizontally scaled control plane).

A device's WebSocket lives on ONE worker, but the task driving it may run on another, and
the user's approval/answer/pause may arrive on a third. Each worker keeps its sockets and
live tasks in memory; this module connects them through Redis:

- Device directory: `cl:dev:<id>` -> "<worker>|<kind>" (TTL, heartbeat-refreshed) plus
  up/down notices broadcast to every live worker's inbox. `DeviceRegistry.get()` returns a `RemoteDeviceSession` proxy
  for a device connected elsewhere, so the graph/controller code is unchanged.
- Worker RPC: one inbox LIST per worker (`cl:inbox:<worker>`, read with BLPOP), request/
  reply correlated by id. Lists rather than pub/sub: nothing is lost while a reader
  reconnects. Liveness is a TTL key per worker, so a call to a worker that shut down
  fails FAST (one that crashed fails within WORKER_TTL_S).
- Task directory: `cl:task:<id>` -> worker running it. Control calls (approve, answer,
  pause, resume, cancel, trace, approvals) go to that worker.
- Task snapshots (`cl:snap:<id>` + an index) for reads on any worker, and event fan-out
  to every worker so `/ws/events` on any worker sees every task.

Without Redis, `LocalCluster` keeps the single-worker behaviour (everything is local).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import time
import uuid
from typing import Any

log = logging.getLogger("agent.cluster")

DEV_TTL_S = 30
TASK_TTL_S = 120
SNAP_TTL_S = 7 * 86400
HOME_TTL_S = 86400
ORPHAN_GRACE_S = 30     # a non-terminal snapshot this stale + no live claim = dead
WORKER_TTL_S = 30
INBOX_TTL_S = 60
HEARTBEAT_S = 10
SNAP_INDEX_MAX = 2000

# Control operations a worker will run for its peers (on tasks it is running).
CONTROL_OPS = ("resolve_approval", "decide_approval", "get_approval", "approvals_for",
               "resolve_answer", "pause", "resume", "cancel", "trace")


class RemoteError(RuntimeError):
    """A peer worker is gone or the remote call failed (treated like a disconnect)."""


class LocalCluster:
    """Single worker: no routing. Same interface as RedisCluster."""
    enabled = False
    worker_id = "local"

    async def start(self, app_state) -> None: ...
    async def stop(self) -> None: ...
    async def device_up(self, session) -> None: ...
    async def device_down(self, session) -> None: ...
    async def kick(self, device_id: str) -> bool:
        return False
    def claim_task(self, task_id: str) -> None: ...
    def runs_here(self, task_id: str) -> bool:
        return True
    def release_task(self, task_id: str) -> None: ...
    async def task_worker(self, task_id: str) -> str | None:
        return None
    async def task_alive(self, task_id: str) -> bool:
        return True
    async def control(self, worker: str, op: str, **args) -> Any:
        raise RemoteError("no cluster")
    def note_approval(self, approval_id: str, task_id: str) -> None: ...
    async def approval_task(self, approval_id: str) -> str | None:
        return None
    def put_snapshot(self, task_json: dict) -> None: ...
    async def get_snapshot(self, task_id: str) -> dict | None:
        return None
    async def list_snapshots(self, limit: int = 200) -> list[dict]:
        return []
    def publish_event(self, event: dict) -> None: ...


class RemoteDeviceSession:
    """Proxy for a device whose socket is on another worker (DeviceSession interface)."""

    def __init__(self, cluster: "RedisCluster", device_id: str, kind: str, worker: str):
        self.cluster = cluster
        self.device_id = device_id
        self.kind = kind
        self.worker = worker
        self.remote = True

    async def request(self, type_: str, timeout: float = 60.0, **data: Any) -> dict:
        return await self.cluster.call(self.worker, "dev.request", timeout=timeout + 5,
                                       device_id=self.device_id, type=type_,
                                       req_timeout=timeout, data=data)

    async def push(self, type_: str, **data: Any) -> None:
        await self.cluster.cast(self.worker, "dev.push", device_id=self.device_id,
                                type=type_, data=data)

    def on_message(self, msg: dict) -> None:
        pass

    def fail_all(self, exc: Exception) -> None:
        pass


class RedisCluster:
    enabled = True

    def __init__(self, redis_client, worker_id: str | None = None):
        self.r = redis_client
        self.worker_id = worker_id or f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:4]}"
        self._calls: dict[str, asyncio.Future] = {}
        self._tasks: set[asyncio.Task] = set()
        self._running: set[str] = set()          # task ids this worker is driving
        self._peers: set[str] = set()            # other live workers (broadcast targets)
        self._reader: asyncio.Task | None = None
        self._beat: asyncio.Task | None = None
        self._stopping = False
        self.devices = None
        self.store = None
        self.controller = None

    @classmethod
    def from_url(cls, url: str) -> "RedisCluster":
        import redis.asyncio as redis
        return cls(redis.from_url(url, decode_responses=True))

    # ------------------------------------------------------------------ #
    # transport: one inbox LIST per worker (BLPOP), liveness via a TTL key.
    # Lists (not pub/sub) so a message isn't lost while a reader reconnects.
    # ------------------------------------------------------------------ #
    @staticmethod
    def _inbox(worker: str) -> str:
        return f"cl:inbox:{worker}"

    @staticmethod
    def _alive_key(worker: str) -> str:
        return f"cl:worker:{worker}"

    async def _send(self, worker: str, msg: dict) -> None:
        if not await self.r.exists(self._alive_key(worker)):
            raise RemoteError(f"worker {worker} is offline")
        async with self.r.pipeline(transaction=False) as p:
            p.rpush(self._inbox(worker), json.dumps({**msg, "from": self.worker_id}, default=str))
            p.expire(self._inbox(worker), INBOX_TTL_S)   # a dead worker's inbox can't grow
            await p.execute()

    async def _broadcast(self, msg: dict) -> None:
        if not self._peers:
            return
        data = json.dumps({**msg, "from": self.worker_id}, default=str)
        async with self.r.pipeline(transaction=False) as p:
            for w in self._peers:
                p.rpush(self._inbox(w), data)
                p.expire(self._inbox(w), INBOX_TTL_S)
            await p.execute()

    async def _refresh_peers(self) -> None:
        peers = set()
        async for key in self.r.scan_iter(match="cl:worker:*", count=200):
            w = key[len("cl:worker:"):]
            if w != self.worker_id:
                peers.add(w)
        self._peers = peers

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    async def start(self, app_state) -> None:
        self.devices = app_state.devices
        self.store = app_state.store
        self.controller = app_state.controller
        self.devices.cluster = self
        await self.r.set(self._alive_key(self.worker_id), "1", ex=WORKER_TTL_S)
        self._reader = asyncio.create_task(self._read_loop())
        await self._refresh_peers()
        await self._broadcast({"op": "hello"})            # peers add us immediately
        await self._resync()
        self._beat = asyncio.create_task(self._heartbeat())
        log.info("cluster: worker %s online (%d peer(s), %d remote device(s))",
                 self.worker_id, len(self._peers), len(self.devices._remote))

    async def stop(self) -> None:
        self._stopping = True
        for t in (self._beat, self._reader):
            if t is not None:
                t.cancel()
        try:
            for sid in list(self.devices._sessions) if self.devices else []:
                await self._withdraw(sid)
            await self.r.delete(self._alive_key(self.worker_id), self._inbox(self.worker_id))
            await self._broadcast({"op": "bye"})
        except Exception as e:
            log.warning("cluster stop: %s", e)
        for fut in self._calls.values():
            if not fut.done():
                fut.set_exception(RemoteError("worker stopping"))

    def _spawn(self, coro) -> None:
        t = asyncio.create_task(coro)
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)

    # ------------------------------------------------------------------ #
    # device directory
    # ------------------------------------------------------------------ #
    async def device_up(self, session) -> None:
        kind = getattr(session, "kind", "phone")
        await self.r.set(f"cl:dev:{session.device_id}", f"{self.worker_id}|{kind}", ex=DEV_TTL_S)
        self.devices._remote.pop(session.device_id, None)
        await self._broadcast({"op": "dev.up", "device_id": session.device_id, "kind": kind})

    async def device_down(self, session) -> None:
        # Only if no newer session for the same device took over on this worker.
        if self.devices.get_local(session.device_id) is None:
            await self._withdraw(session.device_id)

    async def _withdraw(self, device_id: str) -> None:
        key = f"cl:dev:{device_id}"
        if (await self.r.get(key) or "").split("|")[0] == self.worker_id:
            await self.r.delete(key)
            await self._broadcast({"op": "dev.down", "device_id": device_id})

    async def kick(self, device_id: str) -> bool:
        """Close a device's socket wherever it is (revocation)."""
        await self._broadcast({"op": "dev.kick", "device_id": device_id})
        return bool(self._peers)

    async def _resync(self) -> None:
        """Rebuild the remote-device map from the directory (covers missed notices and
        workers that died without saying goodbye — their keys expire)."""
        remote: dict[str, RemoteDeviceSession] = {}
        async for key in self.r.scan_iter(match="cl:dev:*", count=200):
            val = await self.r.get(key)
            if not val:
                continue
            worker, _, kind = val.partition("|")
            did = key[len("cl:dev:"):]
            if worker != self.worker_id and self.devices.get_local(did) is None:
                remote[did] = RemoteDeviceSession(self, did, kind or "phone", worker)
        self.devices._remote = remote

    async def _heartbeat(self) -> None:
        while True:
            await asyncio.sleep(HEARTBEAT_S)
            try:
                async with self.r.pipeline(transaction=False) as p:
                    p.set(self._alive_key(self.worker_id), "1", ex=WORKER_TTL_S)
                    for did, s in list(self.devices._sessions.items()):
                        if getattr(s, "kind", "") in ("phone", "desktop"):
                            p.set(f"cl:dev:{did}", f"{self.worker_id}|{s.kind}", ex=DEV_TTL_S)
                    for tid in self._running:
                        p.set(f"cl:task:{tid}", self.worker_id, ex=TASK_TTL_S)
                    await p.execute()
                await self._refresh_peers()
                await self._resync()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("cluster heartbeat failed: %s", e)

    # ------------------------------------------------------------------ #
    # task directory, approvals, snapshots, events
    # ------------------------------------------------------------------ #
    def claim_task(self, task_id: str) -> None:
        self._running.add(task_id)
        self._spawn(self.r.set(f"cl:task:{task_id}", self.worker_id, ex=TASK_TTL_S))

    def runs_here(self, task_id: str) -> bool:
        return task_id in self._running

    def release_task(self, task_id: str) -> None:
        # The task is done, but its approval records + trace stay in THIS worker's memory:
        # keep a 'home' pointer so peers can still read them.
        self._running.discard(task_id)

        async def release():
            async with self.r.pipeline(transaction=False) as p:
                p.delete(f"cl:task:{task_id}")
                p.set(f"cl:home:{task_id}", self.worker_id, ex=HOME_TTL_S)
                await p.execute()
        self._spawn(release())

    async def task_alive(self, task_id: str) -> bool:
        """True while some worker is driving the task (heartbeat-refreshed key)."""
        return bool(await self.r.exists(f"cl:task:{task_id}"))

    async def task_worker(self, task_id: str) -> str | None:
        """The worker running the task, else the one that last ran it."""
        return (await self.r.get(f"cl:task:{task_id}")
                or await self.r.get(f"cl:home:{task_id}"))

    def note_approval(self, approval_id: str, task_id: str) -> None:
        self._spawn(self.r.set(f"cl:appr:{approval_id}", task_id, ex=SNAP_TTL_S))

    async def approval_task(self, approval_id: str) -> str | None:
        return await self.r.get(f"cl:appr:{approval_id}")

    def put_snapshot(self, task_json: dict) -> None:
        async def write():
            async with self.r.pipeline(transaction=False) as p:
                p.set(f"cl:snap:{task_json['id']}",
                      json.dumps({**task_json, "snap_ts": time.time()}, default=str),
                      ex=SNAP_TTL_S)
                p.zadd("cl:snaps", {task_json["id"]: float(task_json.get("created_at") or 0)})
                p.zremrangebyrank("cl:snaps", 0, -SNAP_INDEX_MAX - 1)
                await p.execute()
        self._spawn(write())

    async def get_snapshot(self, task_id: str) -> dict | None:
        raw = await self.r.get(f"cl:snap:{task_id}")
        return json.loads(raw) if raw else None

    async def list_snapshots(self, limit: int = 200) -> list[dict]:
        ids = await self.r.zrevrange("cl:snaps", 0, limit - 1)
        if not ids:
            return []
        raws = await self.r.mget([f"cl:snap:{i}" for i in ids])
        return [json.loads(x) for x in raws if x]

    def publish_event(self, event: dict) -> None:
        if self._peers:
            self._spawn(self._broadcast({"op": "event", "event": event}))

    # ------------------------------------------------------------------ #
    # RPC
    # ------------------------------------------------------------------ #
    async def call(self, worker: str, op: str, timeout: float = 30.0, **args) -> Any:
        corr = uuid.uuid4().hex
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._calls[corr] = fut
        try:
            await self._send(worker, {"op": op, "corr": corr, **args})
            return await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError:
            raise RemoteError(f"worker {worker} did not answer {op} in {timeout:.0f}s")
        finally:
            self._calls.pop(corr, None)

    async def cast(self, worker: str, op: str, **args) -> None:
        await self._send(worker, {"op": op, **args})

    async def control(self, worker: str, op: str, **args) -> Any:
        if op not in CONTROL_OPS:
            raise ValueError(op)
        return await self.call(worker, "task.ctl", timeout=15.0, ctl=op, args=args)

    async def _reply(self, msg: dict, *, result: Any = None, error: str = "") -> None:
        if msg.get("corr") and msg.get("from"):
            try:
                await self._send(msg["from"], {"op": "reply", "corr": msg["corr"],
                                               "result": result, "error": error})
            except RemoteError:
                pass                                    # caller went away

    async def _read_loop(self) -> None:
        inbox = self._inbox(self.worker_id)
        while not self._stopping:
            try:
                item = await self.r.blpop([inbox], timeout=1)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("cluster read failed: %s", e)
                await asyncio.sleep(1.0)
                continue
            if not item:
                continue
            try:
                msg = json.loads(item[1])
            except (TypeError, ValueError):
                continue
            op = msg.get("op")
            sender = msg.get("from", "")
            if sender and sender != self.worker_id:
                self._peers.add(sender)
            if op == "reply":
                fut = self._calls.get(msg.get("corr", ""))
                if fut is not None and not fut.done():
                    if msg.get("error"):
                        fut.set_exception(RemoteError(msg["error"]))
                    else:
                        fut.set_result(msg.get("result"))
            elif op == "event":
                self.store.publish_local(msg.get("event") or {})
            elif op == "hello":
                pass                                    # sender recorded above
            elif op == "bye":
                self._peers.discard(sender)
                for did, s in list(self.devices._remote.items()):
                    if s.worker == sender:
                        self.devices._remote.pop(did, None)
            elif op in ("dev.up", "dev.down", "dev.kick"):
                self._spawn(self._on_bcast(msg))
            else:
                self._spawn(self._on_call(msg))      # never block the reader (handoffs)

    async def _on_bcast(self, msg: dict) -> None:
        did = msg.get("device_id", "")
        op = msg.get("op")
        if op == "dev.up":
            local = self.devices.get_local(did)
            if local is not None and getattr(local, "socket", None) is not None:
                # The device reconnected elsewhere: our socket is stale.
                self.devices.remove(did, local)
                try:
                    await local.socket.close(code=1012)
                except Exception:
                    pass
            proxy = RemoteDeviceSession(self, did, msg.get("kind", "phone"), msg["from"])
            self.devices._remote[did] = proxy
            # Re-deliver any prompt a task HERE is blocked on (it reconnected elsewhere).
            await self.controller.on_device_connected(proxy)
        elif op == "dev.down":
            cur = self.devices._remote.get(did)
            if cur is not None and cur.worker == msg.get("from"):
                self.devices._remote.pop(did, None)
        elif op == "dev.kick":
            local = self.devices.get_local(did)
            if local is not None and getattr(local, "socket", None) is not None:
                try:
                    await local.socket.close(code=1008)
                except Exception:
                    pass

    async def _on_call(self, msg: dict) -> None:
        op = msg.get("op")
        try:
            if op == "dev.request":
                s = self.devices.get_local(msg["device_id"])
                if s is None:
                    raise RemoteError(f"device {msg['device_id']} is not on this worker")
                res = await s.request(msg["type"], timeout=float(msg.get("req_timeout") or 60),
                                      **(msg.get("data") or {}))
                await self._reply(msg, result=res)
            elif op == "dev.push":
                s = self.devices.get_local(msg["device_id"])
                if s is not None:
                    await s.push(msg["type"], **(msg.get("data") or {}))
            elif op == "task.ctl":
                await self._reply(msg, result=control_local(self.controller, msg["ctl"], msg.get("args") or {}))
            else:
                await self._reply(msg, error=f"unknown op {op}")
        except Exception as e:
            await self._reply(msg, error=f"{type(e).__name__}: {e}")

def control_local(c, op: str, a: dict) -> Any:
    """Run a control op on THIS worker's controller (JSON-safe result)."""
    if op == "resolve_approval":
        return c.resolve_approval(a["task_id"], bool(a["approved"]), a.get("note", ""),
                                  approver=a.get("approver", "device"))
    if op == "decide_approval":
        ok, res = c.decide_approval(a["approval_id"], bool(a["approved"]),
                                    a.get("note", ""), approver=a.get("approver", ""))
        return [ok, res]
    if op == "get_approval":
        rec = c.get_approval(a["approval_id"])
        return rec.view() if rec else None
    if op == "approvals_for":
        return [r.view() for r in c.approvals_for(a["task_id"])]
    if op == "resolve_answer":
        return c.resolve_answer(a["task_id"], a.get("facts") or {})
    if op == "pause":
        return c.pause(a["task_id"])
    if op == "resume":
        return c.resume(a["task_id"])
    if op == "cancel":
        c.cancel(a["task_id"])
        return True
    if op == "trace":
        from . import observability as obs
        return obs.STORE.timeline(a["task_id"])
    raise ValueError(op)


def make_cluster(redis_url: str, enabled: bool = True):
    if redis_url.strip() and enabled:
        return RedisCluster.from_url(redis_url.strip())
    return LocalCluster()
