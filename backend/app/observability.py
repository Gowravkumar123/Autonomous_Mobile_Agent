"""
Observability (blueprint: "OpenTelemetry + Langfuse — trace model calls, graph nodes, tool
calls and latency"; "observable execution: every step produces structured status, timing,
logs and evidence").

One instrumentation — OpenTelemetry — three destinations:
  - always: a built-in per-task timeline (GET /v1/tasks/{id}/trace), no backend needed;
  - LANGFUSE_PUBLIC_KEY/SECRET_KEY: Langfuse (ingests OTLP natively; gen_ai.* spans become
    "generations" with token usage);
  - OTEL_ENDPOINT: any OTLP/HTTP backend (Jaeger, Grafana Tempo, Honeycomb, ...).

Span tree per task:
  task -> plan -> llm.chat ...           (intent / planning)
       -> node.observe -> device.observe
       -> node.reason  -> llm.chat
       -> node.act     -> device.act | tool.call | approval.wait
       -> memory.recall

Privacy: prompts, screen contents and screenshots are NOT recorded unless TRACE_CONTENT=true
(then truncated). Token usage + estimated cost are accumulated per task (task.usage) via a
context variable, so they're visible even with no tracing backend at all.
"""
from __future__ import annotations

import base64
import contextvars
import json
import logging
import threading
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Iterator

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor, TracerProvider
from opentelemetry.trace import Status, StatusCode

log = logging.getLogger("agent.observability")

# Approximate USD per 1M tokens (input, output). VERIFY on groq.com/pricing — the free tier
# costs $0; this estimates the paid-tier equivalent. Override with LLM_PRICES_JSON.
DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "openai/gpt-oss-120b": (0.15, 0.75),
    "openai/gpt-oss-20b": (0.10, 0.50),
}

TRACE_CONTENT = False
_prices = dict(DEFAULT_PRICES)
_usage: contextvars.ContextVar[dict | None] = contextvars.ContextVar("task_usage", default=None)


# --------------------------------------------------------------------------- #
# built-in per-task timeline
# --------------------------------------------------------------------------- #
class TaskSpanStore(SpanProcessor):
    """Keeps a compact timeline of recent tasks' spans in memory (bounded)."""

    KEEP_ATTRS = ("task.", "gen_ai.", "llm.", "device.", "tool.", "approval.", "react.",
                  "memory.", "groq.", "error")

    def __init__(self, max_tasks: int = 200, max_spans: int = 600):
        self.max_tasks, self.max_spans = max_tasks, max_spans
        self._trace_task: dict[int, str] = {}
        self._spans: OrderedDict[str, list[dict]] = OrderedDict()
        self._lock = threading.Lock()

    def on_start(self, span, parent_context=None) -> None:
        tid = (span.attributes or {}).get("task.id")
        if tid:
            with self._lock:
                self._trace_task[span.context.trace_id] = str(tid)
                self._spans.setdefault(str(tid), [])
                self._spans.move_to_end(str(tid))
                while len(self._spans) > self.max_tasks:
                    old, _ = self._spans.popitem(last=False)
                    for k in [k for k, v in self._trace_task.items() if v == old]:
                        self._trace_task.pop(k, None)

    def on_end(self, span: ReadableSpan) -> None:
        with self._lock:
            tid = self._trace_task.get(span.context.trace_id)
            if tid is None or tid not in self._spans:
                return
            bucket = self._spans[tid]
            if len(bucket) >= self.max_spans:
                return
            attrs = {k: v for k, v in (span.attributes or {}).items()
                     if k.startswith(self.KEEP_ATTRS)}
            bucket.append({
                "name": span.name,
                "span_id": format(span.context.span_id, "016x"),
                "parent_id": format(span.parent.span_id, "016x") if span.parent else None,
                "start_ns": span.start_time, "end_ns": span.end_time,
                "duration_ms": round((span.end_time - span.start_time) / 1e6, 1),
                "status": span.status.status_code.name,
                "attributes": attrs,
            })

    def timeline(self, task_id: str) -> list[dict]:
        with self._lock:
            spans = list(self._spans.get(task_id, []))
        if not spans:
            return []
        t0 = min(s["start_ns"] for s in spans)
        out = []
        # Same start tick (coarse Windows clock): the longer span (the parent) comes first.
        for s in sorted(spans, key=lambda s: (s["start_ns"], -s["end_ns"])):
            d = dict(s)
            d["start_ms"] = round((d.pop("start_ns") - t0) / 1e6, 1)
            d.pop("end_ns", None)
            out.append(d)
        return out

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


STORE = TaskSpanStore()
_provider: TracerProvider | None = None
_tracer = None
MODE = "local"


def _default_tracer():
    global _provider, _tracer
    if _tracer is None:
        _provider = TracerProvider(resource=Resource.create({"service.name": "onepiece-agent"}))
        _provider.add_span_processor(STORE)
        _tracer = _provider.get_tracer("onepiece")
    return _tracer


def setup(settings) -> str:
    """Configure exporters from settings. Returns the mode: local | langfuse | otlp."""
    global _provider, _tracer, MODE, TRACE_CONTENT, _prices
    TRACE_CONTENT = bool(getattr(settings, "trace_content", False))
    _prices = dict(DEFAULT_PRICES)
    raw = getattr(settings, "llm_prices_json", "") or ""
    if raw.strip():
        try:
            _prices.update({k: (float(v[0]), float(v[1])) for k, v in json.loads(raw).items()})
        except (ValueError, TypeError, IndexError) as e:
            log.warning("LLM_PRICES_JSON ignored: %s", e)

    provider = TracerProvider(resource=Resource.create(
        {"service.name": getattr(settings, "otel_service_name", "onepiece-agent")}))
    provider.add_span_processor(STORE)
    endpoint, headers, MODE = "", {}, "local"
    pk, sk = getattr(settings, "langfuse_public_key", ""), getattr(settings, "langfuse_secret_key", "")
    if pk and sk:
        host = (getattr(settings, "langfuse_host", "") or "https://cloud.langfuse.com").rstrip("/")
        endpoint = f"{host}/api/public/otel/v1/traces"
        headers = {"Authorization": "Basic " + base64.b64encode(f"{pk}:{sk}".encode()).decode()}
        MODE = "langfuse"
    elif getattr(settings, "otel_endpoint", ""):
        endpoint = settings.otel_endpoint
        for pair in (getattr(settings, "otel_headers", "") or "").split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                headers[k.strip()] = v.strip()
        MODE = "otlp"
    if endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        provider.add_span_processor(BatchSpanProcessor(
            OTLPSpanExporter(endpoint=endpoint, headers=headers, timeout=10)))
    if _provider is not None:
        try:
            _provider.shutdown()
        except Exception:
            pass
    _provider, _tracer = provider, provider.get_tracer("onepiece")
    log.info("tracing: %s%s", MODE, f" -> {endpoint}" if endpoint else "")
    return MODE


def add_processor(processor) -> None:
    """Attach an extra span processor (tests: InMemorySpanExporter)."""
    _default_tracer()
    _provider.add_span_processor(processor)


def shutdown() -> None:
    if _provider is not None:
        try:
            _provider.force_flush(5000)
            _provider.shutdown()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# spans
# --------------------------------------------------------------------------- #
def _clean(attrs: dict) -> dict:
    out = {}
    for k, v in attrs.items():
        if v is None:
            continue
        out[k] = v if isinstance(v, (str, bool, int, float)) else str(v)
    return out


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[Any]:
    """Async-safe (contextvars) span. Never raises because of tracing itself."""
    tracer = _tracer or _default_tracer()
    with tracer.start_as_current_span(name, attributes=_clean(attrs),
                                      record_exception=True, set_status_on_exception=True) as sp:
        yield sp


def set_attrs(sp, **attrs: Any) -> None:
    try:
        sp.set_attributes(_clean(attrs))
    except Exception:
        pass


def mark_error(sp, message: str) -> None:
    try:
        sp.set_status(Status(StatusCode.ERROR, message[:200]))
        sp.set_attribute("error.message", message[:300])
    except Exception:
        pass


def content(sp, key: str, text: str, limit: int = 2000) -> None:
    """Record prompt/response text ONLY when TRACE_CONTENT is enabled (privacy default)."""
    if TRACE_CONTENT and text:
        set_attrs(sp, **{key: text[:limit]})


# --------------------------------------------------------------------------- #
# token usage / cost per task
# --------------------------------------------------------------------------- #
def begin_task_usage() -> dict:
    u = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0,
         "est_cost_usd": 0.0, "by_model": {}}
    _usage.set(u)
    return u


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    p = _prices.get(model)
    if p is None:
        return None
    return (input_tokens * p[0] + output_tokens * p[1]) / 1_000_000


def record_llm(model: str, input_tokens: int, output_tokens: int, reasoning_tokens: int = 0) -> None:
    u = _usage.get()
    if u is None:
        return
    u["llm_calls"] += 1
    u["input_tokens"] += input_tokens
    u["output_tokens"] += output_tokens
    u["reasoning_tokens"] += reasoning_tokens
    cost = estimate_cost(model, input_tokens, output_tokens)
    if cost is not None:
        u["est_cost_usd"] = round(u["est_cost_usd"] + cost, 6)
    m = u["by_model"].setdefault(model, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
    m["calls"] += 1
    m["input_tokens"] += input_tokens
    m["output_tokens"] += output_tokens
