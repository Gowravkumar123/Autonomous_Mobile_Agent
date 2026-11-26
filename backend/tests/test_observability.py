"""
Observability: span tree for a real task run, built-in timeline, LLM spans with token usage
and cost (Groq HTTP mocked — offline), privacy default, Langfuse/OTLP config, a real OTLP
export over HTTP to a local receiver, and the /trace + /usage endpoints.

Run:  python -m tests.test_observability
"""
from __future__ import annotations

import asyncio
import base64
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

os.environ["DATABASE_URL"] = ""
os.environ["GROQ_API_KEY"] = ""
os.environ["GEMINI_API_KEY"] = ""
os.environ["EMBEDDING_BACKEND"] = "hashing"
os.environ["AUTH_SECRET"] = "test-secret-" + "o" * 40

import httpx                                                          # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor        # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter)

from app import observability as obs                                  # noqa: E402
from app.agent.controller import AgentController                     # noqa: E402
from app.agent.memory import MemoryManager                            # noqa: E402
from app.agent.state import Task, TaskState                           # noqa: E402
from app.config import Settings                                       # noqa: E402
from app.llm.groq_provider import GroqProvider                        # noqa: E402
from app.llm.router import LLMRouter                                  # noqa: E402
from app.policy.engine import PolicyEngine                            # noqa: E402
from app.store import TaskStore                                       # noqa: E402
from app.ws import DeviceRegistry                                     # noqa: E402
from tests.test_flow import FakeDevice                                # noqa: E402

PASS = 0
PROMPT = ("Book two train tickets from Hyderabad to Bengaluru for tomorrow. Prefer AC 3-tier. "
          "Compare trains, enter passenger details, stop before payment.")


def check(cond: bool, label: str) -> None:
    global PASS
    assert cond, f"FAIL: {label}"
    PASS += 1
    print(f"PASS {label}")


async def span_tree():
    mem = InMemorySpanExporter()
    obs.add_processor(SimpleSpanProcessor(mem))
    store, devices = TaskStore(), DeviceRegistry()
    c = AgentController(LLMRouter(Settings(groq_api_key="", gemini_api_key="")),
                        PolicyEngine(0), devices, store, MemoryManager())
    dev = FakeDevice(c); devices.add(dev)                       # type: ignore[arg-type]
    t = Task(prompt=PROMPT, device_id="sim"); dev.task_id = t.id; store.create(t)
    await c.run_task(t)
    spans = mem.get_finished_spans()
    names = [s.name for s in spans]
    root = next(s for s in spans if s.name == "task")
    check(t.state == TaskState.COMPLETED and root.attributes["task.id"] == t.id,
          "root 'task' span carries task.id")
    check(all(s.context.trace_id == root.context.trace_id for s in spans
              if s.name != "task" and s.attributes.get("task.id") is None),
          "every span of the run is in ONE trace")
    for n in ("plan", "node.observe", "node.validate", "node.reason", "node.act",
              "device.observe", "device.act", "approval.wait"):
        check(n in names, f"span emitted: {n}")
    aw = next(s for s in spans if s.name == "approval.wait")
    check(aw.attributes.get("approval.outcome") == "approved" and aw.attributes["approval.risk"] == 4,
          "approval.wait records risk + outcome (human time separated from agent time)")
    do = next(s for s in spans if s.name == "device.observe")
    check(do.attributes.get("device.elements", 0) > 0, "device.observe records screen size")
    check(root.attributes.get("task.state") == "COMPLETED", "root span records final state")

    tl = obs.STORE.timeline(t.id)
    check(tl and tl[0]["name"] == "task" and tl[0]["start_ms"] == 0.0 and
          any(s["name"] == "node.act" for s in tl), "built-in timeline (no backend needed)")
    mem.clear()


def mocked_groq(responses: list[dict]):
    """Patch httpx.AsyncClient so GroqProvider talks to a canned transport."""
    seq = iter(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=next(seq))
    real = httpx.AsyncClient

    def factory(*a, **kw):
        kw["transport"] = httpx.MockTransport(handler)
        return real(*a, **kw)
    return real, factory


def groq_reply(content: str, finish: str = "stop", pin=120, pout=40, reason=25) -> dict:
    return {"choices": [{"message": {"content": content}, "finish_reason": finish}],
            "usage": {"prompt_tokens": pin, "completion_tokens": pout, "queue_time": 0.02,
                      "completion_tokens_details": {"reasoning_tokens": reason}}}


async def llm_spans_and_usage():
    mem = InMemorySpanExporter()
    obs.add_processor(SimpleSpanProcessor(mem))
    p = GroqProvider("k", "openai/gpt-oss-120b", "qwen/qwen3.8-27b")
    real, factory = mocked_groq([groq_reply('{"ok":1}'),
                                 groq_reply("", "length", 100, 900, 900),   # reasoning ate it
                                 groq_reply('{"ok":2}', "stop", 100, 60, 30)])
    httpx.AsyncClient = factory
    try:
        usage = obs.begin_task_usage()
        await p.complete("sys", "secret user text", json_mode=True)
        await p.complete("sys", "u", json_mode=True)                        # empty -> retry
    finally:
        httpx.AsyncClient = real
    spans = [s for s in mem.get_finished_spans() if s.name == "llm.chat"]
    a = spans[0].attributes
    check(a["gen_ai.system"] == "groq" and a["gen_ai.request.model"] == "openai/gpt-oss-120b"
          and a["gen_ai.usage.input_tokens"] == 120 and a["gen_ai.usage.output_tokens"] == 40,
          "llm.chat span has GenAI semantic-convention usage (Langfuse 'generation')")
    check(a["gen_ai.usage.reasoning_tokens"] == 25 and a["groq.queue_time_ms"] == 20.0,
          "reasoning tokens + Groq queue time recorded")
    b = spans[1].attributes
    check(b.get("llm.retry.empty_length") and b["llm.requests"] == 2
          and b["gen_ai.usage.output_tokens"] == 960, "empty-answer retry is visible and BOTH "
          "billable requests are counted")
    check(usage["llm_calls"] == 3 and usage["input_tokens"] == 320
          and usage["output_tokens"] == 1000 and usage["reasoning_tokens"] == 955,
          "per-task usage accumulates every request")
    expect = (320 * 0.15 + 1000 * 0.75) / 1e6
    check(abs(usage["est_cost_usd"] - expect) < 1e-9, f"estimated cost = ${expect:.6f}")
    check("input.value" not in a and "output.value" not in a,
          "privacy: prompts/completions NOT traced by default")
    obs.TRACE_CONTENT = True
    httpx.AsyncClient = mocked_groq([groq_reply("hello")])[1]
    try:
        await p.complete("sys", "visible prompt")
    finally:
        httpx.AsyncClient = real
        obs.TRACE_CONTENT = False
    last = [s for s in mem.get_finished_spans() if s.name == "llm.chat"][-1].attributes
    check("visible prompt" in last.get("input.value", "") and last.get("output.value") == "hello",
          "TRACE_CONTENT=true records (truncated) prompt + completion")


def langfuse_config():
    import opentelemetry.exporter.otlp.proto.http.trace_exporter as mod
    captured = {}

    class Fake:
        def __init__(self, endpoint=None, headers=None, timeout=None):
            captured.update(endpoint=endpoint, headers=headers)

        def export(self, spans):
            from opentelemetry.sdk.trace.export import SpanExportResult
            return SpanExportResult.SUCCESS

        def shutdown(self):
            pass

        def force_flush(self, timeout_millis=None):
            return True
    real, mod.OTLPSpanExporter = mod.OTLPSpanExporter, Fake
    try:
        mode = obs.setup(SimpleNamespace(langfuse_public_key="pk-lf-1", langfuse_secret_key="sk-lf-2",
                                         langfuse_host="https://cloud.langfuse.com/",
                                         trace_content=False, llm_prices_json=""))
    finally:
        mod.OTLPSpanExporter = real
    check(mode == "langfuse" and captured["endpoint"] ==
          "https://cloud.langfuse.com/api/public/otel/v1/traces", "Langfuse OTLP endpoint")
    check(captured["headers"]["Authorization"] ==
          "Basic " + base64.b64encode(b"pk-lf-1:sk-lf-2").decode(), "Langfuse basic-auth header")


def otlp_over_the_wire():
    got = []

    class Recv(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            got.append((self.path, self.headers.get("Content-Type"), len(body),
                        self.headers.get("x-team")))
            self.send_response(200); self.end_headers()

        def log_message(self, *a):
            pass
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Recv)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    mode = obs.setup(SimpleNamespace(otel_endpoint=f"http://127.0.0.1:{srv.server_address[1]}/v1/traces",
                                     otel_headers="x-team=onepiece", trace_content=False,
                                     llm_prices_json="", otel_service_name="onepiece-test"))
    with obs.span("task", **{"task.id": "wire-test"}):
        with obs.span("tool.call", **{"tool.name": "demo.add"}):
            pass
    obs.shutdown()                                    # flushes the batch processor
    for _ in range(40):
        if got:
            break
        time.sleep(0.1)
    srv.shutdown()
    check(mode == "otlp" and got and got[0][0] == "/v1/traces"
          and got[0][1] == "application/x-protobuf" and got[0][2] > 50 and got[0][3] == "onepiece",
          "spans exported over OTLP/HTTP (protobuf) with custom headers")
    obs.setup(SimpleNamespace(trace_content=False, llm_prices_json=""))   # back to local


def http_endpoints():
    from app.config import get_settings
    get_settings.cache_clear()
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as cl:
        admin = {"Authorization": f"Bearer {app.state.auth.issue_admin()}"}
        dev = FakeDevice(app.state.controller); app.state.devices.add(dev)   # type: ignore
        tid = cl.post("/tasks", headers=admin, json={"prompt": PROMPT, "device_id": "sim"}).json()["task_id"]
        dev.task_id = tid
        for _ in range(150):
            if cl.get(f"/tasks/{tid}", headers=admin).json()["state"] in ("COMPLETED", "FAILED"):
                break
            time.sleep(0.1)
        tr = cl.get(f"/v1/tasks/{tid}/trace", headers=admin).json()
        names = {s["name"] for s in tr["spans"]}
        check({"task", "plan", "node.act", "device.observe", "approval.wait"} <= names
              and tr["total_ms"] and "device" in tr["time_by_kind_ms"],
              "GET /v1/tasks/{id}/trace: timeline + time split by model/device/approval")
        check(cl.get("/v1/usage", headers=admin).status_code == 200, "GET /v1/usage (admin)")
        check(cl.get(f"/v1/tasks/{tid}/trace").status_code == 401, "trace requires auth")
        check(cl.get("/health").json()["tracing"] == "local", "/health reports tracing mode")


async def main():
    await span_tree()
    await llm_spans_and_usage()
    langfuse_config()
    otlp_over_the_wire()


if __name__ == "__main__":
    asyncio.run(main())
    http_endpoints()
    print(f"\nOBSERVABILITY PASS  ({PASS} checks)")
