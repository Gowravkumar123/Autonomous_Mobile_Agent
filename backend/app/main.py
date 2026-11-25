"""FastAPI entrypoint — REST control plane + device/event WebSockets.

Every route except /health and device pairing requires a bearer token (see app/auth):
admin tokens see everything; a device token is bound to one device_id and can only
drive that device's socket and its own tasks.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import (Depends, FastAPI, File, HTTPException, Request, Response, UploadFile,
                     WebSocket, WebSocketDisconnect)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .agent.controller import AgentController
from .agent.memory import MemoryManager
from .agent.state import TERMINAL, Task
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

    # Authentication (needs the DB for the device registry, so it comes after).
    app.state.auth = AuthManager(s.auth_secret, app.state.db, s.auth_token_days)
    await app.state.auth.load()
    if not app.state.auth.enabled:
        log.warning("!!! AUTH DISABLED — AUTH_SECRET is not set; every endpoint is OPEN. "
                    "Generate one: python -m app.auth.cli new-secret")

    log.info("Agent backend ready. LLM: %s | DB: %s | auth: %s", app.state.llm.active,
             "postgres" if app.state.db else "in-memory",
             "on" if app.state.auth.enabled else "OFF")
    yield
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


def _own_task(p: Principal, task_id: str) -> Task:
    task = app.state.store.get(task_id)
    if task is None or not _visible(p, task):
        raise HTTPException(404, "task not found")
    return task


def _task_view(task: Task, full: bool = False) -> dict:
    view = {
        "task_id": task.id, "state": task.state.value, "prompt": task.prompt,
        "device_id": task.device_id, "owner": task.owner, "executor": task.executor,
        "evidence": task.evidence,
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
            "auth": "on" if app.state.auth.enabled else "OFF"}
    if optional_principal(request) is not None:          # details only for signed-in callers
        body["devices"] = list(app.state.devices._sessions.keys())
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
        code, ttl = app.state.auth.start_pairing()
    except AuthError as e:
        raise HTTPException(400, str(e))
    return {"code": code, "expires_in": ttl}


@app.get("/v1/devices")
async def list_devices(_: Principal = Depends(admin)):
    online = set(app.state.devices._sessions.keys())
    return [{**d, "online": d["device_id"] in online} for d in app.state.auth.list_devices()]


@app.post("/v1/devices/{device_id}/revoke")
async def revoke_device(device_id: str, _: Principal = Depends(admin)):
    ok = await app.state.auth.revoke(device_id)
    session = app.state.devices.get(device_id)
    if session is not None:                       # kick its live socket immediately
        try:
            await session.socket.close(code=1008)
        except Exception:
            pass
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
    if executor not in ("auto", "mobile", "browser", ""):
        raise HTTPException(422, "executor must be auto, mobile or browser")
    if body.start_url and not body.start_url.startswith(("http://", "https://")):
        raise HTTPException(422, "start_url must be an http(s) URL")
    task = Task(prompt=body.prompt, device_id=device_id, owner=p.sub,
                executor="" if executor == "auto" else executor, start_url=body.start_url)
    app.state.store.create(task)
    asyncio.create_task(app.state.controller.run_task(task))
    return {"task_id": task.id, "state": task.state.value}


@app.get("/v1/tasks")
@app.get("/tasks")
async def list_tasks(p: Principal = Depends(principal)):
    return [_task_view(t) for t in app.state.store.all() if _visible(p, t)]


@app.get("/v1/tasks/{task_id}")
@app.get("/tasks/{task_id}")
async def get_task(task_id: str, p: Principal = Depends(principal)):
    return _task_view(_own_task(p, task_id), full=True)


@app.post("/tasks/{task_id}/approve")
async def approve(task_id: str, body: Approve, p: Principal = Depends(principal)):
    """Legacy: decide the task's current pending approval."""
    _own_task(p, task_id)
    return {"ok": app.state.controller.resolve_approval(task_id, body.approved, body.note,
                                                        approver=p.sub)}


@app.post("/v1/tasks/{task_id}/pause")
async def pause_task(task_id: str, p: Principal = Depends(principal)):
    _own_task(p, task_id)
    return {"ok": app.state.controller.pause(task_id)}


@app.post("/v1/tasks/{task_id}/resume")
async def resume_task(task_id: str, p: Principal = Depends(principal)):
    _own_task(p, task_id)
    return {"ok": app.state.controller.resume(task_id)}


@app.get("/v1/tasks/{task_id}/evidence/{name}")
async def task_evidence(task_id: str, name: str, p: Principal = Depends(principal)):
    """Final-state screenshot captured by the browser executor (blueprint: evidence)."""
    from pathlib import Path
    from fastapi.responses import FileResponse
    task = _own_task(p, task_id)
    if name not in task.evidence:                    # only files recorded for THIS task
        raise HTTPException(404, "evidence not found")
    path = Path(__file__).resolve().parents[1] / "evidence" / Path(name).name
    if not path.is_file():
        raise HTTPException(404, "evidence file missing")
    return FileResponse(path, media_type="image/png")


@app.get("/v1/tasks/{task_id}/approvals")
async def task_approvals(task_id: str, p: Principal = Depends(principal)):
    _own_task(p, task_id)
    return [r.view() for r in app.state.controller.approvals_for(task_id)]


@app.get("/v1/approvals/{approval_id}")
async def get_approval(approval_id: str, p: Principal = Depends(principal)):
    rec = app.state.controller.get_approval(approval_id)
    if rec is None:
        raise HTTPException(404, "approval not found")
    _own_task(p, rec.task_id)
    return rec.view()


@app.post("/v1/approvals/{approval_id}/decision")
async def decide_approval(approval_id: str, body: Approve, p: Principal = Depends(principal)):
    """Approve or reject one EXACT pending action (blueprint: transaction-bound approval)."""
    rec = app.state.controller.get_approval(approval_id)
    if rec is None:
        raise HTTPException(404, "approval not found")
    _own_task(p, rec.task_id)
    ok, result = app.state.controller.decide_approval(approval_id, body.approved, body.note,
                                                      approver=p.sub)
    if not ok:
        raise HTTPException(409, result)
    return {"approval_id": approval_id, "decision": result}


@app.post("/v1/tasks/{task_id}/answer")
@app.post("/tasks/{task_id}/answer")
async def answer(task_id: str, body: Answer, p: Principal = Depends(principal)):
    _own_task(p, task_id)
    return {"ok": app.state.controller.resolve_answer(task_id, body.facts)}


@app.post("/v1/tasks/{task_id}/cancel")
@app.post("/tasks/{task_id}/cancel")
async def cancel(task_id: str, p: Principal = Depends(principal)):
    _own_task(p, task_id)
    app.state.controller.cancel(task_id)
    return {"ok": True}


@app.post("/stop")
async def stop_all(p: Principal = Depends(principal)):
    """Emergency stop: cancel every unfinished task this caller may control."""
    stopped = 0
    for t in app.state.store.all():
        if t.state not in TERMINAL and _visible(p, t):
            app.state.controller.cancel(t.id)
            stopped += 1
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
    controller: AgentController = app.state.controller
    # Re-deliver any prompt a waiting task is blocked on (survives ws flaps).
    asyncio.create_task(controller.on_device_connected(session))
    try:
        while True:
            msg = await socket.receive_json()
            mtype = msg.get("type")
            if mtype in ("approval_response", "answer"):
                task = app.state.store.get(msg.get("task_id", ""))
                if task is None or not _visible(p, task):
                    continue                  # can't answer someone else's task
                if mtype == "approval_response":
                    controller.resolve_approval(task.id, bool(msg.get("approved")),
                                                msg.get("note", ""), approver=p.sub)
                else:
                    controller.resolve_answer(task.id, msg.get("facts", {}))
            else:
                session.on_message(msg)   # correlate observe/act responses by rid
    except WebSocketDisconnect:
        pass
    finally:
        session.fail_all(RuntimeError("device disconnected"))
        app.state.devices.remove(device_id, session)   # only if still the current one


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
        task = app.state.store.get(task_id)
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
                task = app.state.store.get(event.get("task_id", ""))
                if task is None or not _visible(p, task):
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
