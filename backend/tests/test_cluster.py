"""
Multi-worker routing (app/cluster.py).

1. In-process: two worker stacks (controller + registry + store) on ONE fake Redis. The
   phone is connected to worker B; the task runs on worker A. Every observe/act, the
   approval, events, snapshots, trace and pause/cancel cross workers; prompts are
   re-delivered when the phone reconnects elsewhere; a dead worker fails fast.
2. Real processes: TWO uvicorn servers sharing a TCP fake Redis. The simulated phone
   connects over WebSocket to server B, the task is created on server A (mock LLM), and
   every read/control call is made against B.

Run:  python -m tests.test_cluster
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

os.environ["DATABASE_URL"] = ""
os.environ["GROQ_API_KEY"] = ""
os.environ["GEMINI_API_KEY"] = ""

import fakeredis                                          # noqa: E402

from app.agent.controller import AgentController          # noqa: E402
from app.agent.memory import MemoryManager                # noqa: E402
from app.agent.state import Task, TaskState               # noqa: E402
from app.cluster import RedisCluster, RemoteError         # noqa: E402
from app.config import Settings                           # noqa: E402
from app.llm.router import LLMRouter                      # noqa: E402
from app.policy.engine import PolicyEngine                # noqa: E402
from app.store import TaskStore                           # noqa: E402
from app.ws import DeviceRegistry                         # noqa: E402
from simulator.mock_device import IrctcPhone              # noqa: E402

PROMPT = ("Book two train tickets from Hyderabad to Bengaluru for tomorrow. "
          "Prefer AC 3-tier. Compare trains, enter passenger details, stop before payment.")
PASS = 0


def check(cond: bool, label: str) -> None:
    global PASS
    assert cond, f"FAIL: {label}"
    PASS += 1
    print(f"PASS {label}")


async def until(pred, timeout: float = 10.0, step: float = 0.05) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        await asyncio.sleep(step)
    return pred()


# --------------------------------------------------------------------------- #
# 1. In-process: two workers, one Redis
# --------------------------------------------------------------------------- #
async def worker(server, name: str):
    store, devices = TaskStore(), DeviceRegistry()
    c = AgentController(LLMRouter(Settings(groq_api_key="", gemini_api_key="")),
                        PolicyEngine(spend_threshold=0), devices, store, MemoryManager())
    cl = RedisCluster(fakeredis.aioredis.FakeRedis(server=server, decode_responses=True),
                      worker_id=name)
    store.cluster = c.cluster = cl
    await cl.start(SimpleNamespace(devices=devices, store=store, controller=c))
    return SimpleNamespace(store=store, devices=devices, c=c, cl=cl)


class SimPhone:
    """A phone whose socket lives on one worker; answers prompts like the real app does
    (its replies arrive on ITS worker and must be routed to the task's worker)."""
    kind = "phone"

    def __init__(self, w, device_id="sim"):
        self.w, self.device_id = w, device_id
        self.phone = IrctcPhone()
        self.requests: list[str] = []
        self.pushes: list[tuple[str, dict]] = []

    async def request(self, type_: str, timeout: float = 60.0, **data):
        self.requests.append(type_)
        if type_ == "observe":
            return self.phone.observe()
        if type_ == "act":
            return self.phone.act(data.get("action", {}))
        if type_ == "handoff":
            return self.phone.handoff()
        return {}

    async def push(self, type_: str, **data):
        self.pushes.append((type_, data))
        if type_ == "approval_request":
            asyncio.get_running_loop().create_task(self.reply(
                "resolve_approval", task_id=data["task_id"], approved=True, note="sim"))

    async def reply(self, op: str, **args):
        """What the ws endpoint does with an inbound reply: route to the task's worker."""
        cl = self.w.cl
        target = await cl.task_worker(args["task_id"])
        await cl.control(target, op, **args)

    def on_message(self, msg): ...
    def fail_all(self, exc): ...


async def connect(w, phone: SimPhone) -> None:
    w.devices.add(phone)
    await w.cl.device_up(phone)


async def in_process() -> None:
    server = fakeredis.FakeServer()
    A = await worker(server, "A")
    B = await worker(server, "B")
    phone = SimPhone(B)
    await connect(B, phone)
    check(await until(lambda: A.devices.get("sim") is not None),
          "worker A learns the phone on worker B (directory broadcast)")
    proxy = A.devices.get("sim")
    check(getattr(proxy, "remote", False) and proxy.worker == "B" and proxy.kind == "phone",
          "A resolves it to a remote proxy (same DeviceSession interface)")
    check(A.devices.get_local("sim") is None and B.devices.get_local("sim") is phone,
          "the socket itself stays on B")

    events_on_b = B.store.subscribe()
    task = Task(prompt=PROMPT, device_id="sim", owner="admin")
    A.store.create(task)
    await asyncio.wait_for(A.c.run_task(task), 120)
    check(task.state == TaskState.COMPLETED, f"task on A completed driving the phone on B "
                                             f"({task.state.value})")
    check(phone.requests.count("observe") >= 3 and "act" in phone.requests,
          "every observe/act was routed A -> B -> phone")
    recs = A.c.approvals_for(task.id)
    check(recs and recs[0].decision == "approved" and recs[0].approver == "device",
          "the approval answered on B was routed back to A")
    got = []
    while not events_on_b.empty():
        got.append(events_on_b.get_nowait())
    check(any(e["task_id"] == task.id and e["kind"] == "state" for e in got) and
          all(e.get("owner") == "admin" for e in got if e["task_id"] == task.id),
          "A's task events fan out to B's subscribers (with owner for filtering)")
    await asyncio.sleep(0.2)
    snap = await B.cl.get_snapshot(task.id)
    check(snap and snap["state"] == "COMPLETED" and snap["history"],
          "B reads the task's snapshot")
    check(any(s["id"] == task.id for s in await B.cl.list_snapshots()), "listed via the index")
    check(await B.cl.task_worker(task.id) == "A" and not await B.cl.task_alive(task.id),
          "finished task keeps a 'home' pointer to A, but isn't alive")
    spans = await B.cl.control("A", "trace", task_id=task.id)
    check(any(s["name"] == "task" for s in spans), "B fetches A's trace over RPC")
    views = await B.cl.control("A", "approvals_for", task_id=task.id)
    check(views and views[0]["decision"] == "approved", "B fetches A's approval records")
    check(await B.cl.approval_task(recs[0].approval_id) == task.id,
          "approval id -> task id resolvable from any worker")

    # ---- pause / resume / cancel from B on a task running on A ----
    slow = SimPhone(B, "slow")

    async def slow_observe(type_, timeout=60.0, **data):
        await asyncio.sleep(0.3)
        return await SimPhone.request(slow, type_, timeout, **data)
    slow.request = slow_observe
    await connect(B, slow)
    await until(lambda: A.devices.get("slow") is not None)
    t2 = Task(prompt=PROMPT, device_id="slow", owner="admin")
    A.store.create(t2)
    run2 = asyncio.create_task(A.c.run_task(t2))
    await until(lambda: t2.state == TaskState.RUNNING, 30)
    check(await B.cl.task_alive(t2.id) and await B.cl.task_worker(t2.id) == "A",
          "running task is claimed by A in the directory")
    check(await B.cl.control("A", "pause", task_id=t2.id) is True, "pause from B")
    check(await until(lambda: t2.state == TaskState.PAUSED, 20), "...task on A pauses")
    await B.cl.control("A", "resume", task_id=t2.id)
    check(await until(lambda: t2.state != TaskState.PAUSED, 10), "resume from B")
    await B.cl.control("A", "cancel", task_id=t2.id)
    await asyncio.wait_for(run2, 30)
    check(t2.state == TaskState.CANCELLED, "cancel from B stops the task on A")

    # ---- prompt re-delivery when the phone reconnects on ANOTHER worker ----
    ask = asyncio.create_task(A.c._ask_user(Task(id="t-ask", device_id="late"),
                                            ["Which class?"], [["AC", "Sleeper"]]))
    await asyncio.sleep(0.2)
    A.store._tasks["t-ask"] = Task(id="t-ask", device_id="late",
                                   state=TaskState.WAITING_FOR_USER)
    A.cl.claim_task("t-ask")                 # (run_task does this for real tasks)
    late = SimPhone(B, "late")
    await connect(B, late)                                   # phone comes back on B
    check(await until(lambda: any(t == "ask_user" for t, _ in late.pushes), 5),
          "pending question re-delivered to the phone after it reconnected on B")
    await late.reply("resolve_answer", task_id="t-ask", facts={"clarification_answer": "AC"})
    facts = await asyncio.wait_for(ask, 5)
    check(facts.get("clarification_answer") == "AC", "its answer (on B) resumed the ask on A")

    # ---- dead worker: B goes away ----
    await B.cl.stop()
    check(await until(lambda: A.devices.get("sim") is None, 5),
          "B's shutdown withdraws its devices from A's directory")
    t0 = time.monotonic()
    try:
        await proxy.request("observe")
        ok = False
    except RemoteError as e:
        ok = "offline" in str(e)
    check(ok and time.monotonic() - t0 < 1.0, "a call to a dead worker fails FAST (no timeout)")
    await A.cl.stop()


# --------------------------------------------------------------------------- #
# 2. Two real server processes
# --------------------------------------------------------------------------- #
def free_port() -> int:
    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        return so.getsockname()[1]


def spawn(port: int, env: dict) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1",
         "--port", str(port), "--log-level", "warning"],
        cwd=str(Path(__file__).resolve().parents[1]), env=env,
        stdout=subprocess.DEVNULL,
        stderr=open(Path(os.environ.get("TEMP", ".")) / f"cluster-worker-{port}.log", "w"))


async def wait_http(base: str) -> None:
    import httpx
    async with httpx.AsyncClient() as h:
        for _ in range(300):
            try:
                if (await h.get(base + "/health")).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.2)
    raise RuntimeError(f"{base} did not start")


async def phone_client(url: str, token: str, stats: dict, stop: asyncio.Event) -> None:
    """The phone app's protocol, backed by the IRCTC simulator."""
    import websockets
    phone = IrctcPhone()
    async with websockets.connect(url, additional_headers={"Authorization": f"Bearer {token}"},
                                  max_size=8 << 20) as ws:
        stats["connected"] = True
        while not stop.is_set():
            try:
                raw = await asyncio.wait_for(ws.recv(), 0.5)
            except asyncio.TimeoutError:
                continue
            msg = json.loads(raw)
            t, rid = msg.get("type"), msg.get("rid")
            stats[t] = stats.get(t, 0) + 1
            if t == "observe":
                payload = phone.observe()
            elif t == "act":
                payload = phone.act(msg.get("action") or {})
            elif t == "handoff":
                payload = phone.handoff()
            elif t == "approval_request":
                await ws.send(json.dumps({"type": "approval_response",
                                          "task_id": msg["task_id"], "approved": True,
                                          "note": "approved on the phone"}))
                continue
            else:
                continue
            await ws.send(json.dumps({"rid": rid, "payload": payload}))


async def two_processes() -> None:
    import httpx
    import websockets
    rport = free_port()
    redis_srv = fakeredis.TcpFakeServer(("127.0.0.1", rport), server_type="redis")
    threading.Thread(target=redis_srv.serve_forever, daemon=True).start()
    secret = "test-secret-" + "c" * 40
    env = {**os.environ, "REDIS_URL": f"redis://127.0.0.1:{rport}/0", "AUTH_SECRET": secret,
           "DATABASE_URL": "", "EMBEDDING_BACKEND": "hashing", "GROQ_API_KEY": "",
           "GEMINI_API_KEY": "", "MCP_CONFIG_PATH": "does-not-exist.json",
           "MEMORY_LEARN": "false", "BROWSER_ENABLED": "false"}
    pa, pb = free_port(), free_port()
    A, B = f"http://127.0.0.1:{pa}", f"http://127.0.0.1:{pb}"
    procs = [spawn(pa, env), spawn(pb, env)]
    stop = asyncio.Event()
    phone_task = None
    try:
        await asyncio.gather(wait_http(A), wait_http(B))
        os.environ["AUTH_SECRET"] = secret
        from app.auth.manager import AuthManager
        admin_tok = AuthManager(secret).issue_admin()
        adm = {"Authorization": f"Bearer {admin_tok}"}
        async with httpx.AsyncClient(timeout=30) as h:
            code = (await h.post(A + "/v1/devices/pair/start", headers=adm)).json()["code"]
            r = await h.post(B + "/v1/devices/pair", json={"code": code, "device_id": "phone-1"})
            check(r.status_code == 200, "pairing code issued on A is redeemable on B (shared KV)")
            token = r.json()["token"]

            stats: dict = {}
            phone_task = asyncio.create_task(phone_client(
                B.replace("http", "ws") + "/ws/device/phone-1", token, stats, stop))
            await until(lambda: stats.get("connected"), 10)
            ok = False
            for _ in range(50):
                hA = (await h.get(A + "/health", headers=adm)).json()
                if "phone-1" in hA.get("devices", []):
                    ok = True
                    break
                await asyncio.sleep(0.1)
            check(ok, "server A sees the phone connected to server B")

            ev_seen: list[dict] = []

            async def watch_events():
                async with websockets.connect(B.replace("http", "ws") + "/ws/events",
                                              additional_headers=adm) as ws:
                    while True:
                        ev_seen.append(json.loads(await ws.recv()))
            watcher = asyncio.create_task(watch_events())
            await asyncio.sleep(0.5)

            r = await h.post(A + "/v1/tasks", headers=adm,
                             json={"prompt": PROMPT, "device_id": "phone-1"})
            tid = r.json()["task_id"]
            state = ""
            for _ in range(600):
                state = (await h.get(f"{B}/v1/tasks/{tid}", headers=adm)).json().get("state")
                if state in ("COMPLETED", "FAILED", "CANCELLED"):
                    break
                await asyncio.sleep(0.2)
            if state != "COMPLETED":
                v = (await h.get(f"{B}/v1/tasks/{tid}", headers=adm)).json()
                print("    error:", v.get("error"), "| history:", v.get("history"), "| phone:", stats)
                print("    events:", [e["detail"][:100] for e in v.get("events", [])[-8:]])
            check(state == "COMPLETED", f"task created on A, polled on B, completed ({state})")
            check(stats.get("observe", 0) >= 3 and stats.get("act", 0) >= 1 and
                  stats.get("approval_request") == 1,
                  "A drove the phone through B's socket, approval pushed to it")
            view = (await h.get(f"{B}/v1/tasks/{tid}", headers=adm)).json()
            check(view["history"] and view["owner"] == "admin", "B serves the full task view")
            listed = (await h.get(B + "/v1/tasks", headers=adm)).json()
            check(any(t["task_id"] == tid for t in listed), "B's task list includes A's task")
            appr = (await h.get(f"{B}/v1/tasks/{tid}/approvals", headers=adm)).json()
            check(appr and appr[0]["decision"] == "approved" and appr[0]["approver"] == "phone-1",
                  "approval answered on B's socket was decided on A (routed)")
            one = (await h.get(f"{B}/v1/approvals/{appr[0]['approval_id']}", headers=adm))
            check(one.status_code == 200 and one.json()["task_id"] == tid,
                  "GET /v1/approvals/{id} on B (lookup + RPC to A)")
            again = await h.post(f"{B}/v1/approvals/{appr[0]['approval_id']}/decision",
                                 headers=adm, json={"approved": True})
            check(again.status_code == 409, "re-deciding via B is refused (already approved)")
            tr = (await h.get(f"{B}/v1/tasks/{tid}/trace", headers=adm)).json()
            check(tr["spans"] and any(s["name"] == "llm.chat" or s["name"] == "plan"
                                      for s in tr["spans"]), "B returns A's trace timeline")
            await asyncio.sleep(0.3)
            check(any(e.get("task_id") == tid and e.get("kind") == "state" for e in ev_seen),
                  "event stream on B shows A's task live")
            watcher.cancel()

            # Device-scoped visibility still holds across workers.
            other = (await h.post(A + "/v1/devices/pair/start", headers=adm)).json()["code"]
            tok2 = (await h.post(A + "/v1/devices/pair",
                                 json={"code": other, "device_id": "phone-2"})).json()["token"]
            r = await h.get(f"{B}/v1/tasks/{tid}", headers={"Authorization": f"Bearer {tok2}"})
            check(r.status_code == 404, "another device can't read the task on any worker")

            # Worker A dies: B still answers reads from the snapshot.
            procs[0].kill()
            procs[0].wait(10)
            r = await h.get(f"{B}/v1/tasks/{tid}", headers=adm)
            check(r.status_code == 200 and r.json()["state"] == "COMPLETED",
                  "after A dies, B still serves the finished task")
            r = await h.post(f"{B}/v1/tasks/{tid}/cancel", headers=adm)
            check(r.status_code == 503, "control call to a dead worker -> 503 (not a hang)")
    finally:
        stop.set()
        if phone_task is not None:
            phone_task.cancel()
        for p in procs:
            if p.poll() is None:
                p.kill()
            p.wait(10)
        redis_srv.shutdown()
        redis_srv.server_close()


async def main() -> None:
    await in_process()
    await two_processes()
    print(f"\nCLUSTER PASS  ({PASS} checks)")


if __name__ == "__main__":
    asyncio.run(main())
