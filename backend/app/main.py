"""FastAPI entrypoint — REST control plane + device/event WebSockets.

Every route except /health and device pairing requires a bearer token (see app/auth):
admin tokens see everything; a device token is bound to one device_id and can only
drive that device's socket and its own tasks.
"""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import (Depends, FastAPI, File, HTTPException, Request, Response, UploadFile,
                     WebSocket, WebSocketDisconnect)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .agent.controller import AgentController
from .agent.memory import MemoryManager
from .agent.state import TERMINAL, Task, TaskState
from .auth import AuthError, AuthManager, Principal
from .auth.deps import admin, optional_principal, principal, ws_principal
from .config import get_settings
from .llm.router import LLMRouter
from .policy.engine import PolicyEngine
from .skills import registry
from .store import TaskStore
from .ws import DeviceRegistry, DeviceSession

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("agent.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    app.state.settings = s
    from . import observability as obs
    app.state.tracing = obs.setup(s)
    app.state.store = TaskStore()
    app.state.devices = DeviceRegistry()
    app.state.llm = LLMRouter(s)
    app.state.policy = PolicyEngine(spend_threshold=s.spend_approval_threshold)
    app.state.memory = MemoryManager()
    app.state.controller = AgentController(
        app.state.llm, app.state.policy, app.state.devices,
        app.state.store, app.state.memory,
    )
    # Durable persistence (Postgres / Neon). Optional: without DATABASE_URL, or if the
    # DB is unreachable, everything still runs in-memory.
    app.state.db = None
    if s.database_url.strip():
        from .audit.logger import set_audit_db
        from .db.postgres import Database
        db = Database(s.database_url.strip())
        if await db.init():
            app.state.db = db
            app.state.store.db = db
            set_audit_db(db)
            n = await app.state.store.hydrate()
            log.info("Restored %d task(s) from Postgres", n)

    # Long-term memory (pgvector on Neon). The open-source embedding model is loaded once
    # here (first deploy downloads ~130 MB into embedding_cache_dir).
    app.state.memory = None
    if s.memory_enabled:
        from .agent.memory import make_embedder   # MemoryManager is imported at module top
        embedder = await make_embedder(s.embedding_backend, s.embedding_model,
                                       s.embedding_cache_dir, s.embedding_api_url,
                                       s.embedding_api_key)
        app.state.memory = MemoryManager(app.state.db, embedder, ttl_days=s.memory_ttl_days,
                                         include_all_below=s.memory_include_all_below)
        await app.state.memory.init()
        log.info("Memory: %s store, embedder %s",
                 "postgres/pgvector" if app.state.memory.persistent else "in-memory",
                 embedder.name)
    app.state.controller.memory = app.state.memory

    # Tool layer: built-in research tools + every MCP server in the config file.
    from .tools import ToolRegistry
    app.state.tools = ToolRegistry.from_file(
        s.mcp_config_path, builtins=s.tools_builtin, timeout_s=s.tool_timeout_s,
        rate_per_minute=s.tool_rate_per_minute, max_calls_per_task=s.tool_max_calls_per_task)
    await app.state.tools.start()
    app.state.controller.tools = app.state.tools
    log.info("Tools: %s", app.state.tools.status())

    # Shared short-lived state: Redis when REDIS_URL is set (multi-worker), else in-process.
    from .kv import make_kv
    app.state.kv = await make_kv(s.redis_url)
    app.state.tools.kv = app.state.kv
    app.state.controller.kv = app.state.kv
    # Multi-worker routing (device sockets / task control / events) over the same Redis.
    from .cluster import make_cluster
    # Only route through Redis if it actually answered (make_kv falls back to memory).
    app.state.cluster = make_cluster(s.redis_url if app.state.kv.name == "redis" else "",
                                     s.cluster_enabled)
    app.state.store.cluster = app.state.cluster
    app.state.controller.cluster = app.state.cluster

    # Authentication (needs the DB for the device registry, so it comes after).
    app.state.auth = AuthManager(s.auth_secret, app.state.db, s.auth_token_days)
    app.state.auth.kv = app.state.kv
    await app.state.auth.load()
    if not app.state.auth.enabled:
        log.warning("!!! AUTH DISABLED — AUTH_SECRET is not set; every endpoint is OPEN. "
                    "Generate one: python -m app.auth.cli new-secret")

    # Compile the ReAct graph NOW: built lazily it cost the first task after every restart
    # ~6s (LangGraph import + compile) — found via the trace timeline.
    _ = app.state.controller.graph
    await app.state.cluster.start(app.state)

    # Voice gateway: sessions drive the SAME task runtime as text (blueprint 4).
    from .voice.manager import VoiceManager
    app.state.voice = VoiceManager(_voice_backend, db=app.state.db)

    log.info("Agent backend ready. LLM: %s | DB: %s | auth: %s", app.state.llm.active,
             "postgres" if app.state.db else "in-memory",
             "on" if app.state.auth.enabled else "OFF")
    yield
    await app.state.voice.close_all()
    await app.state.cluster.stop()   # withdraw this worker's devices from the directory
    await app.state.tools.stop()     # stop MCP server processes / sessions
    await app.state.kv.close()
    obs.shutdown()                   # flush pending spans to the exporter
    if app.state.db:
        await app.state.db.close()   # flush pending writes on shutdown


app = FastAPI(title="OnePiece Autonomous Agent", version="0.2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _visible(p: Principal, task: Task) -> bool:
    # Visibility follows the OWNER (a phone still owns a task it started even when that
    # task runs in the server-side browser). Older tasks fall back to device_id.
    owner = task.owner or task.device_id
    return p.is_admin or (p.scope == "device" and owner == p.sub)


async def _lookup(task_id: str) -> Task | None:
    """This worker's live copy if it RUNS the task, else the freshest cluster snapshot
    (the task may be running on another worker), else whatever this worker has."""
    task = app.state.store.get(task_id)
    cl = app.state.cluster
    if cl.enabled and not cl.runs_here(task_id):
        snap = await cl.get_snapshot(task_id)
        if snap:
            try:
                remote = Task(**snap)
            except Exception:
                return task
            from .cluster import ORPHAN_GRACE_S
            stale = time.time() - float(snap.get("snap_ts") or 0) > ORPHAN_GRACE_S
            if remote.state not in TERMINAL and stale and not await cl.task_alive(task_id):
                # Its worker died mid-task (heartbeat key expired): don't show it running.
                remote.state = TaskState.FAILED
                remote.error = remote.error or "the worker running this task went offline"
            return remote
    return task


async def _own_task(p: Principal, task_id: str) -> Task:
    task = await _lookup(task_id)
    if task is None or not _visible(p, task):
        raise HTTPException(404, "task not found")
    return task


async def _all_tasks() -> list[Task]:
    """Tasks known to this worker plus every worker's snapshots (newest first)."""
    by_id = {t.id: t for t in app.state.store.all()}
    cl = app.state.cluster
    if cl.enabled:
        for snap in await cl.list_snapshots():
            if not cl.runs_here(snap.get("id", "")):
                try:
                    by_id[snap["id"]] = Task(**snap)
                except Exception:
                    continue
    return sorted(by_id.values(), key=lambda t: t.created_at, reverse=True)


READ_OPS = ("get_approval", "approvals_for", "trace")


async def _ctl(task_id: str, op: str, **args):
    """Run a control op where the task RUNS: here, or on the worker that owns it."""
    from .cluster import RemoteError, control_local
    args["task_id"] = task_id
    cl = app.state.cluster
    if cl.enabled and not cl.runs_here(task_id):
        worker = await cl.task_worker(task_id)
        if worker and worker != cl.worker_id:
            try:
                return await cl.control(worker, op, **args)
            except RemoteError as e:
                if op not in READ_OPS:
                    raise HTTPException(503, f"task's worker unavailable: {e}")
                # reads fall back to whatever this worker knows
    return control_local(app.state.controller, op, args)


async def _approval_task_id(approval_id: str) -> str | None:
    rec = app.state.controller.get_approval(approval_id)
    if rec is not None:
        return rec.task_id
    return await app.state.cluster.approval_task(approval_id)


def _task_view(task: Task, full: bool = False) -> dict:
    view = {
        "task_id": task.id, "state": task.state.value, "prompt": task.prompt,
        "device_id": task.device_id, "owner": task.owner, "executor": task.executor,
        "evidence": task.evidence, "tool_results": task.tool_results, "usage": task.usage,
        "goal": task.goal, "skill": task.skill, "app_hint": task.app_hint,
        "facts": task.facts, "plan": task.plan, "approval_boundary": task.approval_boundary,
        "summary": task.summary, "error": task.error, "step_id": task.step_id,
        "pending_approval": task.pending_approval_reason or None,
    }
    if full:
        view["history"] = task.history
        view["events"] = [e.model_dump() for e in task.events]
    return view


# --------------------------------------------------------------------------- #
# Public
# --------------------------------------------------------------------------- #
@app.get("/health")
async def health(request: Request):
    from .vision.ocr import available as ocr_available
    body = {"ok": True, "llm": app.state.llm.active, "ocr": ocr_available(),
            "db": "postgres" if app.state.db else "in-memory",
            "auth": "on" if app.state.auth.enabled else "OFF",
            "tools": len(app.state.tools.list()),
            "memory": (app.state.memory.embedder.name if app.state.memory else "off"),
            "state": app.state.kv.name, "tracing": app.state.tracing}
    if optional_principal(request) is not None:          # details only for signed-in callers
        body["devices"] = [d.device_id for d in app.state.devices.all()]
    return body


class PairRequest(BaseModel):
    code: str
    device_id: str


@app.post("/v1/devices/pair")
async def pair_device(body: PairRequest):
    """Phone submits the 6-digit code shown to the admin; gets its device token."""
    try:
        token = await app.state.auth.complete_pairing(body.code, body.device_id)
    except AuthError as e:
        status = 429 if "too many" in str(e) else 400
        raise HTTPException(status, str(e))
    return {"device_id": body.device_id, "token": token}


# --------------------------------------------------------------------------- #
# Admin: device management
# --------------------------------------------------------------------------- #
@app.post("/v1/devices/pair/start")
async def pair_start(_: Principal = Depends(admin)):
    try:
        code, ttl = await app.state.auth.start_pairing()
    except AuthError as e:
        raise HTTPException(400, str(e))
    return {"code": code, "expires_in": ttl}


@app.get("/v1/devices")
async def list_devices(_: Principal = Depends(admin)):
    online = {d.device_id for d in app.state.devices.all()}
    return [{**d, "online": d["device_id"] in online} for d in app.state.auth.list_devices()]


@app.post("/v1/devices/{device_id}/revoke")
async def revoke_device(device_id: str, _: Principal = Depends(admin)):
    ok = await app.state.auth.revoke(device_id)
    session = app.state.devices.get_local(device_id)
    if session is not None:                       # kick its live socket immediately
        try:
            await session.socket.close(code=1008)
        except Exception:
            pass
    else:
        await app.state.cluster.kick(device_id)   # its socket is on another worker
    return {"revoked": ok}


# --------------------------------------------------------------------------- #
# Tasks (admin: all; device: its own)
# --------------------------------------------------------------------------- #
class CreateTask(BaseModel):
    prompt: str
    device_id: str = ""
    executor: str = "auto"      # auto | mobile | browser
    start_url: str = ""         # optional first page for the browser executor


class Approve(BaseModel):
    approved: bool
    note: str = ""


class Answer(BaseModel):
    facts: dict = {}


@app.get("/v1/tools")
async def list_tools(p: Principal = Depends(principal)):
    """Tool catalog with each tool's effective policy (allow / approve / deny) and schema.
    There is deliberately NO endpoint to call a tool directly — every call goes through
    a task and the policy gate."""
    tools = [t.brief() for t in app.state.tools.list()]
    if p.is_admin:
        return {"status": app.state.tools.status(), "tools": tools}
    return {"tools": [{k: t[k] for k in ("name", "description", "policy")} for t in tools]}


@app.post("/v1/tools/refresh")
async def refresh_tools(_: Principal = Depends(admin)):
    """Reconnect every MCP server and re-list its tools (after editing the config)."""
    return await app.state.tools.refresh()


# --------------------------------------------------------------------------- #
# Long-term memory (consent: user-approved only; owner-scoped; deletable)
# --------------------------------------------------------------------------- #
class MemoryIn(BaseModel):
    content: str
    kind: str = "preference"


class MemoryQuery(BaseModel):
    query: str


def _mem():
    if app.state.memory is None:
        raise HTTPException(503, "memory is disabled (MEMORY_ENABLED=false)")
    return app.state.memory


@app.get("/v1/memory")
async def list_memory(status: str = "approved", owner: str = "",
                      p: Principal = Depends(principal)):
    if status not in ("approved", "pending", "all"):
        raise HTTPException(422, "status must be approved, pending or all")
    who = (owner or None) if p.is_admin else p.sub      # devices see only their own
    return [m.view() for m in await _mem().list(who, status)]


@app.post("/v1/memory")
async def add_memory(body: MemoryIn, p: Principal = Depends(principal)):
    from .agent.memory import MemoryError_
    try:
        item = await _mem().add(p.sub, body.content, kind=body.kind, source="user",
                                approved=True)
    except MemoryError_ as e:
        raise HTTPException(422, str(e))
    return item.view()


@app.post("/v1/memory/{mem_id}/approve")
async def approve_memory(mem_id: str, p: Principal = Depends(principal)):
    ok = await _mem().approve(mem_id, None if p.is_admin else p.sub)
    if not ok:
        raise HTTPException(404, "memory not found")
    return {"approved": True}


@app.delete("/v1/memory/{mem_id}")
async def delete_memory(mem_id: str, p: Principal = Depends(principal)):
    ok = await _mem().delete(mem_id, None if p.is_admin else p.sub)
    if not ok:
        raise HTTPException(404, "memory not found")
    return {"deleted": True}


@app.post("/v1/memory/search")
async def search_memory(body: MemoryQuery, p: Principal = Depends(principal)):
    return [m.view() for m in await _mem().relevant(p.sub, body.query)]


@app.get("/skills")
async def skills(_: Principal = Depends(principal)):
    return [{"key": s.key, "title": s.title, "apps": s.app_packages,
             "approval_boundary": s.approval_boundary} for s in registry.all_skills()]


@app.post("/v1/tasks")
@app.post("/tasks")
async def create_task(body: CreateTask, p: Principal = Depends(principal)):
    # A device can only start tasks on itself; an admin may target any device.
    device_id = p.sub if p.scope == "device" else body.device_id
    executor = body.executor.strip().lower()
    if executor not in ("auto", "mobile", "browser", "api", "desktop", ""):
        raise HTTPException(422, "executor must be auto, mobile, browser, api or desktop")
    if body.start_url and not body.start_url.startswith(("http://", "https://")):
        raise HTTPException(422, "start_url must be an http(s) URL")
    task = _start_task(body.prompt, owner=p.sub, device_id=device_id,
                       executor="" if executor == "auto" else executor,
                       start_url=body.start_url)
    return {"task_id": task.id, "state": task.state.value}


def _start_task(prompt: str, *, owner: str, device_id: str = "", executor: str = "",
                start_url: str = "") -> Task:
    """The one way tasks start — REST and voice alike."""
    task = Task(prompt=prompt, device_id=device_id, owner=owner, executor=executor,
                start_url=start_url)
    app.state.store.create(task)
    asyncio.create_task(app.state.controller.run_task(task))
    return task


@app.get("/v1/tasks")
@app.get("/tasks")
async def list_tasks(p: Principal = Depends(principal)):
    return [_task_view(t) for t in await _all_tasks() if _visible(p, t)]


@app.get("/v1/tasks/{task_id}")
@app.get("/tasks/{task_id}")
async def get_task(task_id: str, p: Principal = Depends(principal)):
    return _task_view(await _own_task(p, task_id), full=True)


@app.post("/tasks/{task_id}/approve")
async def approve(task_id: str, body: Approve, p: Principal = Depends(principal)):
    """Legacy: decide the task's current pending approval."""
    await _own_task(p, task_id)
    return {"ok": await _ctl(task_id, "resolve_approval", approved=body.approved,
                             note=body.note, approver=p.sub)}


@app.post("/v1/tasks/{task_id}/pause")
async def pause_task(task_id: str, p: Principal = Depends(principal)):
    await _own_task(p, task_id)
    return {"ok": await _ctl(task_id, "pause")}


@app.post("/v1/tasks/{task_id}/resume")
async def resume_task(task_id: str, p: Principal = Depends(principal)):
    await _own_task(p, task_id)
    return {"ok": await _ctl(task_id, "resume")}


@app.get("/v1/tasks/{task_id}/trace")
async def task_trace(task_id: str, p: Principal = Depends(principal)):
    """Built-in timeline of the task's spans (works with no tracing backend): where the
    time went (model vs device vs human approval) and token usage / estimated cost."""
    from . import observability as obs
    task = await _own_task(p, task_id)
    spans = obs.STORE.timeline(task_id) or (await _ctl(task_id, "trace") or [])
    by_kind: dict[str, float] = {}
    for s in spans:
        kind = s["name"].split(".")[0]
        if s["name"] in ("llm.chat", "device.observe", "device.act", "tool.call",
                         "approval.wait", "memory.recall"):
            by_kind[kind] = round(by_kind.get(kind, 0.0) + s["duration_ms"], 1)
    root = next((s for s in spans if s["name"] == "task"), None)
    return {"task_id": task_id, "state": task.state.value, "usage": task.usage,
            "total_ms": root["duration_ms"] if root else None,
            "time_by_kind_ms": by_kind, "spans": spans}


@app.get("/v1/usage")
async def usage_summary(_: Principal = Depends(admin)):
    """Token usage and estimated cost across recent tasks (cost-efficiency view)."""
    totals = {"tasks": 0, "llm_calls": 0, "input_tokens": 0, "output_tokens": 0,
              "reasoning_tokens": 0, "est_cost_usd": 0.0}
    by_model: dict[str, dict] = {}
    per_task = []
    for t in await _all_tasks():
        u = t.usage or {}
        if not u.get("llm_calls"):
            continue
        totals["tasks"] += 1
        for k in ("llm_calls", "input_tokens", "output_tokens", "reasoning_tokens"):
            totals[k] += int(u.get(k, 0))
        totals["est_cost_usd"] = round(totals["est_cost_usd"] + float(u.get("est_cost_usd", 0)), 6)
        for model, m in (u.get("by_model") or {}).items():
            agg = by_model.setdefault(model, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
            for k in agg:
                agg[k] += int(m.get(k, 0))
        per_task.append({"task_id": t.id, "prompt": t.prompt[:80], "state": t.state.value,
                         "llm_calls": u.get("llm_calls"), "est_cost_usd": u.get("est_cost_usd")})
    per_task.sort(key=lambda x: x["est_cost_usd"] or 0, reverse=True)
    return {"totals": totals, "by_model": by_model, "top_tasks": per_task[:20],
            "note": "est_cost_usd is the paid-tier equivalent at approximate list prices; "
                    "Groq's free tier costs $0. Override prices with LLM_PRICES_JSON."}


@app.get("/v1/tasks/{task_id}/evidence/{name}")
async def task_evidence(task_id: str, name: str, p: Principal = Depends(principal)):
    """Final-state screenshot captured by the browser executor (blueprint: evidence)."""
    from pathlib import Path
    from fastapi.responses import FileResponse
    task = await _own_task(p, task_id)
    if name not in task.evidence:                    # only files recorded for THIS task
        raise HTTPException(404, "evidence not found")
    path = Path(__file__).resolve().parents[1] / "evidence" / Path(name).name
    if not path.is_file():
        raise HTTPException(404, "evidence file missing")
    return FileResponse(path, media_type="image/png")


@app.get("/v1/tasks/{task_id}/approvals")
async def task_approvals(task_id: str, p: Principal = Depends(principal)):
    await _own_task(p, task_id)
    return await _ctl(task_id, "approvals_for")


@app.get("/v1/approvals/{approval_id}")
async def get_approval(approval_id: str, p: Principal = Depends(principal)):
    task_id = await _approval_task_id(approval_id)
    if task_id is None:
        raise HTTPException(404, "approval not found")
    await _own_task(p, task_id)
    view = await _ctl(task_id, "get_approval", approval_id=approval_id)
    if view is None:
        raise HTTPException(404, "approval not found")
    return view


@app.post("/v1/approvals/{approval_id}/decision")
async def decide_approval(approval_id: str, body: Approve, p: Principal = Depends(principal)):
    """Approve or reject one EXACT pending action (blueprint: transaction-bound approval)."""
    task_id = await _approval_task_id(approval_id)
    if task_id is None:
        raise HTTPException(404, "approval not found")
    await _own_task(p, task_id)
    ok, result = await _ctl(task_id, "decide_approval", approval_id=approval_id,
                            approved=body.approved, note=body.note, approver=p.sub)
    if not ok:
        raise HTTPException(409, result)
    return {"approval_id": approval_id, "decision": result}


@app.post("/v1/tasks/{task_id}/answer")
@app.post("/tasks/{task_id}/answer")
async def answer(task_id: str, body: Answer, p: Principal = Depends(principal)):
    await _own_task(p, task_id)
    return {"ok": await _ctl(task_id, "resolve_answer", facts=body.facts)}


@app.post("/v1/tasks/{task_id}/cancel")
@app.post("/tasks/{task_id}/cancel")
async def cancel(task_id: str, p: Principal = Depends(principal)):
    await _own_task(p, task_id)
    await _ctl(task_id, "cancel")
    return {"ok": True}


@app.post("/stop")
async def stop_all(p: Principal = Depends(principal)):
    """Emergency stop: cancel every unfinished task this caller may control."""
    stopped = 0
    for t in await _all_tasks():
        if t.state not in TERMINAL and _visible(p, t):
            try:
                await _ctl(t.id, "cancel")
                stopped += 1
            except HTTPException:
                pass                                  # its worker is gone (task is dead)
    return {"stopped": stopped}


# --------------------------------------------------------------------------- #
# Voice (authenticated — otherwise anyone could spend the Groq quota)
# --------------------------------------------------------------------------- #
@app.post("/voice/tts")
async def voice_tts(body: dict, _: Principal = Depends(principal)):
    """Text -> speech (WAV) via Groq PlayAI. 503 if unavailable (phone falls back)."""
    from .voice.groq_audio import tts
    audio = await tts(str(body.get("text", ""))[:1200])
    if not audio:
        return Response(status_code=503)
    return Response(content=audio, media_type="audio/wav")


@app.post("/voice/stt")
async def voice_stt(file: UploadFile = File(...), _: Principal = Depends(principal)):
    """Speech -> text via Groq Whisper."""
    from .voice.groq_audio import stt
    text = await stt(await file.read(), file.filename or "audio.wav")
    return {"text": text}


# --------------------------------------------------------------------------- #
# Voice gateway (blueprint 4 + 9): WS /v1/voice/{session_id} carries 20 ms PCM16 frames
# (binary) both ways plus JSON control/events; WebRTC is the alternative media path.
# --------------------------------------------------------------------------- #
import re as _re                                                    # noqa: E402

_SESSION_ID = _re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _voice_backend(owner: str, device_id: str):
    from .voice.groq_audio import stt_detailed, tts_cached
    from .voice.session import Backend
    s = app.state.settings

    async def create(prompt: str) -> str:
        return _start_task(prompt, owner=owner, device_id=device_id).id

    async def stt(wav: bytes):
        return await stt_detailed(wav, "turn.wav", s.voice_language)

    return Backend(create_task=create, lookup=_lookup, ctl=_ctl,
                   subscribe=app.state.store.subscribe,
                   unsubscribe=app.state.store.unsubscribe,
                   stt=stt, tts=tts_cached,
                   record_turn=lambda t: app.state.voice.record_turn(t))


def _voice_cfg(mode: str):
    from .voice.session import VoiceConfig
    s = app.state.settings
    return VoiceConfig(mode=mode if mode in ("duplex", "ptt") else "duplex",
                       end_silence_ms=s.voice_end_silence_ms,
                       min_confidence=s.voice_min_confidence,
                       progress_every_s=s.voice_progress_s,
                       partial_every_s=s.voice_partial_s, language=s.voice_language)


def _voice_device(p: Principal, requested: str) -> str:
    # A device's voice session drives that device; an admin may name one.
    return p.sub if p.scope == "device" else (requested or "")


class _WsVoiceTransport:
    def __init__(self, socket: WebSocket):
        self.socket = socket
        self._lock = asyncio.Lock()

    async def send_json(self, msg: dict) -> None:
        async with self._lock:
            await self.socket.send_json(msg)

    async def send_audio(self, frame: bytes) -> None:
        async with self._lock:
            await self.socket.send_bytes(frame)


@app.websocket("/v1/voice/{session_id}")
async def ws_voice(socket: WebSocket, session_id: str):
    p = ws_principal(socket)
    if p is None or not _SESSION_ID.match(session_id):
        await socket.close(code=1008)
        return
    q = socket.query_params
    sess = await app.state.voice.get_or_create(
        session_id, p.sub, _voice_device(p, q.get("device_id", "")),
        _voice_cfg(q.get("mode", "duplex")))
    if sess is None:                         # someone else's conversation
        await socket.close(code=1008)
        return
    await socket.accept()
    transport = _WsVoiceTransport(socket)
    await sess.attach(transport)
    try:
        while True:
            msg = await socket.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes"):
                await sess.feed_audio(msg["bytes"])
            elif msg.get("text"):
                try:
                    import json as _json
                    await sess.handle_json(_json.loads(msg["text"]))
                except ValueError:
                    pass
    except WebSocketDisconnect:
        pass
    finally:
        sess.detach(transport)
        app.state.voice.touch(session_id)


class VoiceTurn(BaseModel):
    text: str
    device_id: str = ""


@app.post("/v1/voice/{session_id}/turns")
async def voice_turn(session_id: str, body: VoiceTurn, p: Principal = Depends(principal)):
    """Submit a finalized transcript or a typed fallback turn (same task state)."""
    if not _SESSION_ID.match(session_id):
        raise HTTPException(422, "bad session id")
    sess = await app.state.voice.get_or_create(session_id, p.sub,
                                               _voice_device(p, body.device_id))
    if sess is None:
        raise HTTPException(404, "session not found")
    reply = await sess.submit_text(body.text[:2000])
    return {"session_id": session_id, "reply": reply, "task_id": sess.task_id,
            "pending": sess.pending.kind}


class WebRtcOffer(BaseModel):
    sdp: str
    type: str = "offer"
    mode: str = "duplex"
    device_id: str = ""


@app.post("/v1/voice/{session_id}/webrtc")
async def voice_webrtc(session_id: str, body: WebRtcOffer, p: Principal = Depends(principal)):
    """WebRTC signaling: SDP offer in, answer out. Audio = Opus/SRTP; JSON control and
    events on the "control" data channel."""
    from .voice import webrtc
    if not webrtc.available():
        raise HTTPException(501, "WebRTC unavailable on this server (pip install aiortc)")
    if not _SESSION_ID.match(session_id):
        raise HTTPException(422, "bad session id")
    sess = await app.state.voice.get_or_create(session_id, p.sub,
                                               _voice_device(p, body.device_id),
                                               _voice_cfg(body.mode))
    if sess is None:
        raise HTTPException(404, "session not found")
    ice = [u.strip() for u in app.state.settings.voice_ice_servers.split(",") if u.strip()]
    transport = webrtc.WebRtcTransport(sess, ice)
    answer = await transport.answer(body.sdp, body.type)
    await sess.attach(transport)
    return answer


@app.get("/v1/voice/{session_id}")
async def voice_info(session_id: str, p: Principal = Depends(principal)):
    sess = app.state.voice.sessions.get(session_id)
    if sess is None or (sess.owner != p.sub and not p.is_admin):
        raise HTTPException(404, "session not found")
    return {"session_id": session_id, "owner": sess.owner, "mode": sess.cfg.mode,
            "connected": sess.transport is not None, "task_id": sess.task_id,
            "pending": sess.pending.kind, "turns": sess.turns[-50:]}


# --------------------------------------------------------------------------- #
# WebSockets
# --------------------------------------------------------------------------- #
@app.websocket("/ws/device/{device_id}")
async def ws_device(socket: WebSocket, device_id: str):
    p = ws_principal(socket)
    if p is None or not p.owns_device(device_id):
        await socket.close(code=1008)        # rejected before accept -> HTTP 403
        return
    await socket.accept()
    session = DeviceSession(device_id, socket)
    app.state.devices.add(session)
    await app.state.cluster.device_up(session)     # tasks on other workers can reach it
    controller: AgentController = app.state.controller
    # Re-deliver any prompt a waiting task is blocked on (survives ws flaps).
    asyncio.create_task(controller.on_device_connected(session))
    try:
        while True:
            msg = await socket.receive_json()
            mtype = msg.get("type")
            if mtype in ("approval_response", "answer"):
                task = await _lookup(msg.get("task_id", ""))
                # The owner may answer — and so may the device the task is RUNNING on: the
                # server pushed the prompt to that screen, for the person sitting at it
                # (e.g. an admin-started task on a desktop). Nobody else can.
                if task is None or not (_visible(p, task) or task.device_id == p.sub):
                    continue                  # can't answer someone else's task
                try:                          # routed to the worker running the task
                    if mtype == "approval_response":
                        await _ctl(task.id, "resolve_approval",
                                   approved=bool(msg.get("approved")),
                                   note=msg.get("note", ""), approver=p.sub)
                    else:
                        await _ctl(task.id, "resolve_answer", facts=msg.get("facts", {}))
                except HTTPException as e:
                    log.warning("could not deliver %s for %s: %s", mtype, task.id, e.detail)
            else:
                session.on_message(msg)   # correlate observe/act responses by rid
    except WebSocketDisconnect:
        pass
    finally:
        session.fail_all(RuntimeError("device disconnected"))
        app.state.devices.remove(device_id, session)   # only if still the current one
        await app.state.cluster.device_down(session)


@app.websocket("/ws/events")
async def ws_events(socket: WebSocket):
    await _stream_events(socket, task_id=None)


@app.websocket("/v1/tasks/{task_id}/events")
async def ws_task_events(socket: WebSocket, task_id: str):
    """Blueprint: stream plan updates, tool activity, status and approvals for ONE task."""
    await _stream_events(socket, task_id=task_id)


async def _stream_events(socket: WebSocket, task_id: str | None) -> None:
    p = ws_principal(socket)
    if p is None:
        await socket.close(code=1008)
        return
    if task_id is not None:
        task = await _lookup(task_id)
        if task is None or not _visible(p, task):
            await socket.close(code=1008)
            return
    await socket.accept()
    q = app.state.store.subscribe()
    try:
        while True:
            event = await q.get()
            if task_id is not None and event.get("task_id") != task_id:
                continue
            if not p.is_admin:
                # Events from other workers carry the owner (no local task object).
                task = app.state.store.get(event.get("task_id", ""))
                owner = (task.owner or task.device_id) if task else event.get("owner", "")
                if not (p.scope == "device" and owner and owner == p.sub):
                    continue
            await socket.send_json(event)
    except WebSocketDisconnect:
        pass
    finally:
        app.state.store.unsubscribe(q)


if __name__ == "__main__":
    import uvicorn
    s = get_settings()
    uvicorn.run("app.main:app", host=s.host, port=s.port, reload=False)
