"""
/v1 control plane: pause/resume, transaction-bound approvals (record, expiry, decide by id,
re-validation against a fresh screen), and the HTTP endpoints.

Run:  python -m tests.test_v1
"""
from __future__ import annotations

import asyncio
import os

os.environ["AUTH_SECRET"] = "test-secret-" + "y" * 40
os.environ["DATABASE_URL"] = ""
os.environ["GROQ_API_KEY"] = ""
os.environ["GEMINI_API_KEY"] = ""

from app.agent.approvals import action_binding                    # noqa: E402
from app.agent.controller import AgentController                  # noqa: E402
from app.agent.memory import MemoryManager                        # noqa: E402
from app.agent.state import Task, TaskState                       # noqa: E402
from app.config import Settings                                   # noqa: E402
from app.llm.router import LLMRouter                              # noqa: E402
from app.mobile.schema import (ActionType, AgentAction,           # noqa: E402
                               ScreenObservation, UiElement)
from app.policy.engine import PolicyEngine                        # noqa: E402
from app.store import TaskStore                                   # noqa: E402
from app.ws import DeviceRegistry                                 # noqa: E402
from tests.test_flow import FakeDevice                            # noqa: E402

PROMPT = ("Book two train tickets from Hyderabad to Bengaluru for tomorrow. Prefer AC 3-tier. "
          "Compare trains, enter passenger details, stop before payment.")
PASS = 0


def check(cond: bool, label: str) -> None:
    global PASS
    assert cond, f"FAIL: {label}"
    PASS += 1
    print(f"PASS {label}")


def make(auto_approve: bool = True):
    store, devices = TaskStore(), DeviceRegistry()
    c = AgentController(LLMRouter(Settings(groq_api_key="", gemini_api_key="")),
                        PolicyEngine(spend_threshold=0), devices, store, MemoryManager())
    dev = FakeDevice(c)
    if not auto_approve:
        async def push(type_: str, **data):   # never answers approvals
            return None
        dev.push = push                          # type: ignore[assignment]
    devices.add(dev)                             # type: ignore[arg-type]
    return c, store, dev


async def wait_state(task: Task, states: set, timeout: float = 10.0) -> bool:
    for _ in range(int(timeout / 0.05)):
        if task.state in states:
            return True
        await asyncio.sleep(0.05)
    return False


# --------------------------------------------------------------------------- #
async def test_pause_resume():
    c, store, dev = make()
    task = Task(prompt=PROMPT, device_id="sim"); dev.task_id = task.id; store.create(task)
    c.pause(task.id)                                     # pause before execution starts
    runner = asyncio.create_task(c.run_task(task))
    check(await wait_state(task, {TaskState.PAUSED}), "task pauses at the cycle boundary")
    n = len(task.history)
    await asyncio.sleep(0.6)
    check(len(task.history) == n and task.state == TaskState.PAUSED,
          "no actions are taken while paused")
    c.resume(task.id)
    await asyncio.wait_for(runner, 30)
    check(task.state == TaskState.COMPLETED, "resumed task runs to completion")


async def test_approval_record():
    c, store, dev = make()
    task = Task(prompt=PROMPT, device_id="sim"); dev.task_id = task.id; store.create(task)
    await asyncio.wait_for(c.run_task(task), 30)
    recs = c.approvals_for(task.id)
    check(len(recs) >= 1 and recs[0].decision == "approved" and len(recs[0].action_hash) == 64,
          "approval is a bound record (sha256 action hash) and gets decided")
    check(task.pending_approval_id == "", "pending approval cleared after decision")


async def test_approval_expiry():
    c, store, dev = make(auto_approve=False)
    c.approval_ttl_s = 0.5
    task = Task(prompt=PROMPT, device_id="sim"); dev.task_id = task.id; store.create(task)
    await asyncio.wait_for(c.run_task(task), 30)
    recs = c.approvals_for(task.id)
    check(recs and recs[0].decision == "expired", "unanswered approval expires")
    check(task.state == TaskState.CANCELLED and "expired" in task.events[-1].detail,
          "expired approval stops the sensitive step (task cancelled, not executed)")


async def test_decide_by_id():
    c, store, dev = make(auto_approve=False)
    task = Task(prompt=PROMPT, device_id="sim"); dev.task_id = task.id; store.create(task)
    runner = asyncio.create_task(c.run_task(task))
    for _ in range(200):                                   # wait for the approval to appear
        if task.pending_approval_id:
            break
        await asyncio.sleep(0.05)
    aid = task.pending_approval_id
    check(bool(aid), "task waits on a pending approval id")
    ok, res = c.decide_approval(aid, True, approver="admin")
    check(ok and res == "approved", "decide approval by its exact id")
    ok2, res2 = c.decide_approval(aid, True, approver="admin")
    check(not ok2 and "already" in res2, "an approval can't be decided twice")
    await asyncio.wait_for(runner, 30)
    check(task.state == TaskState.COMPLETED and c.get_approval(aid).approver == "admin",
          "task continues; approver recorded")


async def test_binding_invalidation():
    """Level-3 action approved, then the screen changes before execution -> invalidated."""
    send = ScreenObservation(app_package="com.whatsapp", activity="Chat", elements=[
        UiElement(ref=1, role="Button", text="Send", clickable=True)])
    changed = ScreenObservation(app_package="com.whatsapp", activity="Chat", elements=[
        UiElement(ref=1, role="Button", text="Delete chat", clickable=True)])

    class ScriptedDevice:
        device_id = "sim"

        def __init__(self, ctl):
            self.ctl, self.screen, self.acts = ctl, send, []

        async def request(self, type_, timeout=60.0, **data):
            if type_ == "observe":
                return {"screen": self.screen.model_dump()}
            if type_ == "act":
                self.acts.append(data.get("action", {}))
            return {"ok": True}

        async def push(self, type_, **data):
            if type_ == "approval_request":
                self.screen = changed              # UI shifts while the user is deciding
                tid = data["task_id"]
                asyncio.get_event_loop().call_soon(
                    lambda: self.ctl.resolve_approval(tid, True, "ok"))

    store, devices = TaskStore(), DeviceRegistry()
    c = AgentController(LLMRouter(Settings(groq_api_key="", gemini_api_key="")),
                        PolicyEngine(spend_threshold=0), devices, store, MemoryManager())
    dev = ScriptedDevice(c); devices.add(dev)      # type: ignore[arg-type]
    calls = {"n": 0}

    async def scripted_decide(task, skill, screen, feedback=""):
        calls["n"] += 1
        if calls["n"] == 1:
            return AgentAction(type=ActionType.TAP, ref=1, thought="send it")
        return AgentAction(type=ActionType.FAIL, text="test end")
    c._decide = scripted_decide                    # type: ignore[assignment]

    task = Task(prompt="say hi", device_id="sim", goal="say hi", skill="generic",
                plan=["send"])
    store.create(task)
    await asyncio.wait_for(c._execute_phase(task), 30)
    recs = c.approvals_for(task.id)
    check(recs and recs[0].decision == "invalidated",
          "approval invalidated when the target changed after approval")
    check(not any(a.get("ref") == 1 and a.get("type") == "tap" for a in dev.acts),
          "the changed (unapproved) action was NOT executed")

    a = AgentAction(type=ActionType.TAP, ref=1)
    check(action_binding(a, send) == action_binding(a, send), "binding is deterministic")
    check(action_binding(a, send) != action_binding(a, changed), "binding covers the target label")


def test_http():
    from fastapi.testclient import TestClient
    from app.agent.approvals import ApprovalRecord
    from app.main import app
    import time
    with TestClient(app) as cl:
        auth, ctl, store = app.state.auth, app.state.controller, app.state.store
        admin = {"Authorization": f"Bearer {auth.issue_admin()}"}
        code = cl.post("/v1/devices/pair/start", headers=admin).json()["code"]
        dev = {"Authorization": "Bearer " + cl.post(
            "/v1/devices/pair", json={"code": code, "device_id": "phone-A"}).json()["token"]}

        t = Task(prompt="x", device_id="phone-B"); store.create(t)   # another device's task
        rec = ApprovalRecord(task_id=t.id, step_id=0, action_hash="h" * 64,
                             expires_at=time.time() + 60)
        ctl._approval_records[rec.approval_id] = rec
        url = f"/v1/approvals/{rec.approval_id}/decision"
        check(cl.post(url, json={"approved": True}, headers=dev).status_code == 404,
              "device can't decide another device's approval")
        r = cl.post(url, json={"approved": True}, headers=admin)
        check(r.status_code == 200 and r.json()["decision"] == "approved", "admin decides by id")
        check(cl.post(url, json={"approved": False}, headers=admin).status_code == 409,
              "second decision -> 409")
        check(cl.get(f"/v1/tasks/{t.id}/approvals", headers=admin).json()[0]["decision"]
              == "approved", "GET /v1/tasks/{id}/approvals lists records")
        check(cl.post("/v1/approvals/nope/decision", json={"approved": True},
                      headers=admin).status_code == 404, "unknown approval -> 404")
        check(cl.post(f"/v1/tasks/{t.id}/pause", headers=admin).json()["ok"], "pause endpoint")
        check(cl.post(f"/v1/tasks/{t.id}/resume", headers=admin).json()["ok"], "resume endpoint")
        check(cl.get("/v1/tasks", headers=admin).status_code == 200, "/v1/tasks alias")


async def main():
    await test_pause_resume()
    await test_approval_record()
    await test_approval_expiry()
    await test_decide_by_id()
    await test_binding_invalidation()


if __name__ == "__main__":
    asyncio.run(main())
    test_http()
    print(f"\nV1 PASS  ({PASS} checks)")
