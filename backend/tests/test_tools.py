"""
Tool execution layer: registry (MCP over stdio + HTTP, built-ins), policy resolution,
schema validation, idempotency, rate limits, SSRF guard, and the agent integration
(API executor, approvals bound to exact args, deny, decline, pre-authorization limits).

Run:  python -m tests.test_tools
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

os.environ["DATABASE_URL"] = ""
os.environ["EMBEDDING_BACKEND"] = "hashing"
os.environ["GROQ_API_KEY"] = ""
os.environ["GEMINI_API_KEY"] = ""
os.environ["AUTH_SECRET"] = "test-secret-" + "t" * 40

from app.agent.controller import AgentController                  # noqa: E402
from app.agent.memory import MemoryManager                        # noqa: E402
from app.agent.state import Task, TaskState                       # noqa: E402
from app.config import Settings                                   # noqa: E402
from app.llm.router import LLMRouter                              # noqa: E402
from app.mobile.schema import ActionType, AgentAction             # noqa: E402
from app.policy.engine import PolicyEngine                        # noqa: E402
from app.store import TaskStore                                   # noqa: E402
from app.tools.builtins import _html_to_text                      # noqa: E402
from app.tools.registry import ToolRegistry, _subst_env           # noqa: E402
from app.ws import DeviceRegistry                                 # noqa: E402

PASS = 0
LOCAL = {"command": "python", "args": ["-m", "tests.mcp_test_server"],
         "trust_annotations": True, "tools": {"delete_all_notes": "deny"}}


def check(cond: bool, label: str) -> None:
    global PASS
    assert cond, f"FAIL: {label}"
    PASS += 1
    print(f"PASS {label}")


def free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close()
    return p


# --------------------------------------------------------------------------- #
async def registry_tests(port: int):
    reg = ToolRegistry({"servers": {"local": LOCAL,
                                    "remote": {"url": f"http://127.0.0.1:{port}/mcp"}}},
                       rate_per_minute=1000, max_calls_per_task=100)
    await reg.start()
    try:
        st = reg.status()
        check(st["servers"]["local"]["connected"] and st["servers"]["remote"]["connected"],
              "connects to MCP servers over stdio AND streamable HTTP")
        g = reg.get
        check(g("local.add").policy == "allow" and g("web.search").policy == "allow",
              "trusted read-only + built-in tools auto-allowed")
        check(g("local.create_note").policy == "approve", "write tool needs approval")
        check(g("local.delete_all_notes").policy == "deny", "config can deny a tool")
        check(g("remote.add").policy == "approve",
              "untrusted server: even 'read-only' tools need approval (annotations are hints)")
        check(g("local.pay_invoice").policy == "approve" and g("local.pay_invoice").risk == 4,
              "money-looking tool is risk 4 even on a trusted server")
        fake = SimpleNamespace(name="get_order_status", description="", input_schema={},
                               annotations=SimpleNamespace(read_only_hint=True,
                                                           destructive_hint=False,
                                                           idempotent_hint=True))
        blanket = reg._spec_from_mcp("s", {"default_policy": "allow"}, fake)
        explicit = reg._spec_from_mcp("s", {"tools": {"get_order_status": "allow"}}, fake)
        check(blanket.policy == "approve" and explicit.policy == "allow",
              "blanket rules can't auto-allow money-looking tools; explicit per-tool allow can")

        r = await reg.call(g("local.add"), {"a": 17, "b": 25}, "t1")
        check(r.ok and r.text.strip() == "42", "call + normalize (text result)")
        r = await reg.call(g("remote.get_weather"), {"city": "Hyderabad"}, "t1")
        check(r.ok and json.loads(r.text)["temp_c"] == 31, "remote HTTP tool call works")
        r = await reg.call(g("local.broken"), {}, "t1")
        check(not r.ok and r.error, "tool error surfaces as ok=False")
        for bad, why in (({"a": "x", "b": 1}, "wrong type"), ({"a": 1}, "missing field"),
                         ([1, 2], "not an object")):
            try:
                reg.validate(g("local.add"), bad)
                check(False, f"schema rejects {why}")
            except Exception:
                check(True, f"schema validation rejects args ({why})")

        async def count():   # list results arrive as structured {"result": [...]}
            return len((await reg.call(g("local.list_notes"), {}, "t1")).structured["result"])
        before = await count()
        a1 = await reg.call(g("local.create_note"), {"title": "x", "body": "y"}, "t2")
        a2 = await reg.call(g("local.create_note"), {"title": "x", "body": "y"}, "t2")
        after = await count()
        check(a1.ok and a2.cached and after == before + 1,
              "idempotency: repeated write in a task is NOT re-executed")

        reg.rate_per_minute = 2
        reg._recent.clear()                  # earlier calls were inside the 60s window
        rr = [await reg.call(g("local.add"), {"a": i, "b": 1}, "t3") for i in range(3)]
        check(rr[0].ok and rr[1].ok and not rr[2].ok and "rate limit" in rr[2].error,
              "per-tool rate limit")
        reg.rate_per_minute, reg.max_calls_per_task = 1000, 2
        rb = [await reg.call(g("local.add"), {"a": i, "b": 2}, "t4") for i in range(3)]
        check(not rb[2].ok and "budget" in rb[2].error, "per-task tool-call budget")
        reg.max_calls_per_task = 100

        for url, why in (("http://127.0.0.1:8000/health", "loopback"),
                         ("http://169.254.169.254/latest/meta-data/", "cloud metadata"),
                         ("http://10.0.0.5/", "private network")):
            r = await reg.call(g("web.fetch"), {"url": url}, "t5")
            check(not r.ok and ("non-public" in r.error or "port" in r.error),
                  f"web.fetch SSRF guard blocks {why}")
        try:
            reg.validate(g("web.fetch"), {"url": "file:///etc/passwd"})
            check(False, "web.fetch rejects file://")
        except Exception:
            check(True, "web.fetch rejects non-http(s) URLs at schema validation")
    finally:
        await reg.stop()

    title, text = _html_to_text("<html><title>T &amp; Co</title><script>x()</script>"
                                "<body><p>Hello</p><p>World</p></body></html>")
    check(title == "T & Co" and "Hello" in text and "x()" not in text, "HTML -> readable text")
    os.environ["DEMO_TOKEN"] = "s3cret"
    check(_subst_env({"h": "Bearer ${DEMO_TOKEN}"}) == {"h": "Bearer s3cret"},
          "${ENV} substitution keeps secrets out of the config file")


# --------------------------------------------------------------------------- #
async def agent_tests():
    reg = ToolRegistry({"servers": {"local": LOCAL}}, rate_per_minute=1000)
    await reg.start()

    def make(script):
        store, devices = TaskStore(), DeviceRegistry()
        c = AgentController(LLMRouter(Settings(groq_api_key="", gemini_api_key="")),
                            PolicyEngine(spend_threshold=0), devices, store, MemoryManager())
        c.tools = reg
        it = iter(script)

        async def decide(task, skill, screen, feedback=""):
            return next(it, AgentAction(type=ActionType.FAIL, text="script exhausted"))
        c._decide = decide                                   # type: ignore[assignment]
        return c, store

    def tool(name, **args):
        return AgentAction(type=ActionType.CALL_TOOL, tool=name, args=args)

    async def run(c, store, task, approve: bool | None = None):
        store.create(task)

        async def decider():
            while True:
                if task.pending_approval_id and approve is not None:
                    c.decide_approval(task.pending_approval_id, approve, approver="admin")
                await asyncio.sleep(0.05)
        helper = asyncio.create_task(decider())
        session = await c._select_executor(task)
        try:
            await asyncio.wait_for(c._execute_phase(task), 60)
        finally:
            helper.cancel()
            if session is not None:
                c.devices.remove(session.device_id, session)
        return task

    async def notes() -> int:
        r = await reg.call(reg.get("local.list_notes"), {}, "probe")
        return len(r.structured["result"])

    # 1. pure API executor: two auto-allowed tools, then a verified answer
    c, store = make([tool("local.get_weather", city="Hyderabad"), tool("local.add", a=17, b=25),
                     AgentAction(type=ActionType.DONE, text="31°C and sunny; 17+25=42")])
    t = await run(c, store, Task(prompt="weather + sum", executor="api", owner="admin",
                                 goal="weather and sum", skill="generic"))
    check(t.executor == "api" and t.state == TaskState.COMPLETED and "[OK]" in t.summary
          and all(r["ok"] for r in t.tool_results) and len(t.tool_results) == 2,
          "API executor: tool results -> verified answer (COMPLETED)")

    # 1b. live quirk (server, 2026-11-27): the model nested the call inside "call_tool"
    c, store = make([AgentAction(type=ActionType.CALL_TOOL, tool="call_tool",
                                 args={"tool": "local.add", "args": {"a": 2, "b": 3}}),
                     AgentAction(type=ActionType.DONE, text="2+3=5")])
    t = await run(c, store, Task(prompt="sum", executor="api", owner="admin", goal="sum",
                                 skill="generic"))
    check(t.tool_results and t.tool_results[0]["tool"] == "local.add" and
          t.tool_results[0]["ok"] and t.state == TaskState.COMPLETED,
          "nested call_tool{'tool','args'} is unwrapped to the real tool")
    n_before = await notes()
    c, store = make([AgentAction(type=ActionType.CALL_TOOL, tool="call_tool",
                                 args={"tool": "local.create_note",
                                       "args": {"title": "x", "body": "y"}}),
                     AgentAction(type=ActionType.DONE, text="saved")])
    t = await run(c, store, Task(prompt="note", executor="api", owner="admin", goal="note",
                                 skill="generic"), approve=False)
    check(await notes() == n_before and c.approvals_for(t.id),
          "...and the unwrapped call still needs approval (declined -> not executed)")

    # 2. write tool -> transaction-bound approval -> executes
    n0 = await notes()
    c, store = make([tool("local.create_note", title="Trip", body="sunny"),
                     AgentAction(type=ActionType.DONE, text="saved")])
    t = await run(c, store, Task(prompt="note", executor="api", owner="admin", goal="note",
                                 skill="generic"), approve=True)
    recs = c.approvals_for(t.id)
    check(recs and recs[0].decision == "approved" and t.state == TaskState.COMPLETED
          and await notes() == n0 + 1, "write tool runs only after approval")
    check(len(recs[0].action_hash) == 64, "approval bound to the exact tool + args (hash)")

    # 3. declined approval -> task cancelled, tool NOT executed
    n0 = await notes()
    c, store = make([tool("local.create_note", title="No", body="no")])
    t = await run(c, store, Task(prompt="note", executor="api", owner="admin", goal="note",
                                 skill="generic"), approve=False)
    check(t.state == TaskState.CANCELLED and await notes() == n0,
          "declined approval: tool not executed, task cancelled")

    # 4. denied / unknown / invalid args: refused before dispatch, agent keeps going
    n0 = await notes()
    c, store = make([tool("local.delete_all_notes"), tool("nope.tool"),
                     tool("local.add", a="x"), tool("local.add", a=1, b=2),
                     AgentAction(type=ActionType.DONE, text="3")])
    t = await run(c, store, Task(prompt="x", executor="api", owner="admin", goal="x",
                                 skill="generic"))
    errs = [r["text"] for r in t.tool_results if not r["ok"]]
    check(any("DENIED" in e for e in errs) and await notes() == n0,
          "denied tool is refused (never dispatched)")
    check(any("unknown tool" in e for e in errs), "unknown tool refused with guidance")
    check(any("invalid arguments" in e for e in errs), "invalid args refused before dispatch")
    check(t.state == TaskState.COMPLETED, "agent recovers after refusals and finishes")

    # 5. done without any successful tool result -> unverified
    c, store = make([AgentAction(type=ActionType.DONE, text="probably 42")])
    t = await run(c, store, Task(prompt="x", executor="api", owner="admin", goal="x",
                                 skill="generic"))
    check(t.state == TaskState.WAITING_FOR_USER and "[UNVERIFIED]" in t.summary,
          "'done' with no runtime result is NOT trusted")

    # 6. pre-authorized: risk-3 write runs without asking; risk-4 payment still asks
    c, store = make([tool("local.create_note", title="pre", body="auth"),
                     tool("local.pay_invoice", invoice_id="INV1", amount=500),
                     AgentAction(type=ActionType.DONE, text="ok")])
    t = await run(c, store, Task(prompt="x", executor="api", owner="admin", goal="x",
                                 skill="generic", pre_authorized=True), approve=True)
    recs = c.approvals_for(t.id)
    check(len(recs) == 1 and recs[0].risk_level == 4,
          "pre-authorization skips the risk-3 approval but NEVER the payment (risk 4)")

    # 7. redundant "save it?" confirmation is skipped; the platform approval still happens once
    c, store = make([AgentAction(type=ActionType.ASK_USER, text="Save the note?",
                                 options=["Save the note", "Cancel"]),
                     tool("local.create_note", title="dup", body="confirm"),
                     AgentAction(type=ActionType.DONE, text="saved")])
    t = await run(c, store, Task(prompt="x", executor="api", owner="admin", goal="x",
                                 skill="generic"), approve=True)
    asked = [e for e in t.events if e.kind == "state_change" and e.detail == "needs clarification"]
    check(not asked and len(c.approvals_for(t.id)) == 1 and t.state == TaskState.COMPLETED,
          "go/no-go confirmation skipped; exactly one platform approval")

    # 7b. LIVE regression (found by tracing): a model that ONLY ever asks "Save / Cancel".
    #     Skipped confirmations left no history, so nothing stopped it until the 40-step
    #     cap (42 paid LLM calls). Now: 2 skips, then real questions (capped), then the
    #     stall guard stops it — with a bounded number of model calls.
    calls = {"n": 0}
    c, store = make([])

    async def nagging(task, skill, screen, feedback=""):
        calls["n"] += 1
        return AgentAction(type=ActionType.ASK_USER, text="Save the note?",
                           options=["Save the note", "Cancel"])
    c._decide = nagging                                     # type: ignore[assignment]
    t = Task(prompt="x", executor="api", owner="admin", goal="x", skill="generic")
    store.create(t)

    async def answerer():
        while True:
            fut = c._answers.get(t.id)
            if fut is not None and not fut.done():
                c.resolve_answer(t.id, {"clarification_answer": "Save the note"})
            await asyncio.sleep(0.02)
    helper = asyncio.create_task(answerer())
    session = await c._select_executor(t)
    try:
        await asyncio.wait_for(c._execute_phase(t), 60)
    finally:
        helper.cancel()
        c.devices.remove(session.device_id, session)
    check(t.state == TaskState.WAITING_FOR_USER and "no progress" in t.events[-1].detail
          and calls["n"] <= 12,
          f"ask-loop stopped by the stall guard after {calls['n']} model calls (was 40+)")

    # 8. select_executor auto: intent hint 'api' + tools -> api
    store, devices = TaskStore(), DeviceRegistry()
    c = AgentController(LLMRouter(Settings(groq_api_key="")), PolicyEngine(0), devices,
                        store, MemoryManager())
    c.tools = reg
    check(c._choose_executor(Task(prompt="what's 2+2", executor_hint="api")) == "api",
          "select_executor: intent says tools suffice -> api")
    check(c._choose_executor(Task(prompt="book on bookmyshow.com", executor_hint="api"))
          == "browser", "a named website still wins -> browser")
    await reg.stop()


def http_tests():
    cfg = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump({"servers": {"local": LOCAL}}, cfg); cfg.close()
    os.environ["MCP_CONFIG_PATH"] = cfg.name
    from app.config import get_settings
    get_settings.cache_clear()
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as cl:
        admin = {"Authorization": f"Bearer {app.state.auth.issue_admin()}"}
        body = cl.get("/v1/tools", headers=admin).json()
        names = {t["name"]: t for t in body["tools"]}
        check("local.add" in names and names["local.delete_all_notes"]["policy"] == "deny"
              and "input_schema" in names["local.add"],
              "GET /v1/tools lists tools with effective policy + schema (admin)")
        check(cl.get("/v1/tools").status_code == 401, "/v1/tools requires auth")
        r = cl.post("/v1/tools/refresh", headers=admin).json()
        check(r["servers"]["local"]["connected"], "POST /v1/tools/refresh reconnects")
        check(cl.get("/health").json()["tools"] >= 9, "/health reports tool count")
        check(cl.post("/tasks", headers=admin, json={"prompt": "x", "executor": "api"})
              .status_code == 200, "executor=api accepted")
    os.unlink(cfg.name)



async def web_search_fallback() -> None:
    """Live on the server, DuckDuckGo returned nothing (data-centre IP) and Wikipedia
    403'd the old browser-lookalike User-Agent. Now: DDG first, Wikipedia search second."""
    import httpx
    from app.tools import builtins as bt
    seen_ua = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_ua.append(request.headers.get("user-agent", ""))
        if "duckduckgo" in request.url.host:
            return httpx.Response(202, text="<html>challenge</html>")
        if request.url.host == "en.wikipedia.org":
            return httpx.Response(200, json={"query": {"search": [
                {"title": "Canberra", "snippet": "<span>Canberra</span> is the capital city"}]}})
        return httpx.Response(404)
    r = await bt.web_search({"query": "capital of Australia"}, transport=httpx.MockTransport(handler))
    check(r.ok and "Canberra" in r.text and "results from Wikipedia" in r.text and
          r.structured[0]["url"] == "https://en.wikipedia.org/wiki/Canberra",
          "web.search: DDG blocked -> Wikipedia fallback")
    check(all(ua == bt.UA and not ua.startswith("Mozilla") for ua in seen_ua),
          "web tools send the honest OnePieceAgent User-Agent")

    def ddg_ok(request):
        if "duckduckgo" in request.url.host:
            return httpx.Response(200, text='<a class="result__a" href="https://example.org/x">X</a>'
                                            '<a class="result__snippet">snip</a>')
        raise AssertionError("fallback must not run when DDG has results")
    r = await bt.web_search({"query": "x"}, transport=httpx.MockTransport(ddg_ok))
    check(r.ok and "results from Wikipedia" not in r.text, "web.search: DDG results used first")
    r = await bt.web_search({"query": "x"}, transport=httpx.MockTransport(
        lambda req: httpx.Response(403)))
    check(not r.ok and "no results" in r.error, "web.search: both down -> clear error, no raise")

async def main():
    await web_search_fallback()
    port = free_port()
    srv = subprocess.Popen([sys.executable, "-m", "tests.mcp_test_server", "--http", str(port)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):                                  # wait for the HTTP server
            try:
                socket.create_connection(("127.0.0.1", port), 0.2).close()
                break
            except OSError:
                time.sleep(0.2)
        await registry_tests(port)
        await agent_tests()
    finally:
        srv.terminate()


if __name__ == "__main__":
    asyncio.run(main())
    http_tests()
    print(f"\nTOOLS PASS  ({PASS} checks)")
