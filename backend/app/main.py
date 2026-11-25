"""FastAPI entrypoint — REST control plane + device/event WebSockets."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .agent.controller import AgentController
from .agent.memory import MemoryManager
from .agent.state import Task
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
    log.info("Agent backend ready. Active LLM: %s | DB: %s", app.state.llm.active,
             "postgres" if app.state.db else "in-memory")
    yield
    if app.state.db:
        await app.state.db.close()   # flush pending writes on shutdown


app = FastAPI(title="Autonomous Personal AI Agent", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# --------------------------------------------------------------------------- #
# REST control plane
# --------------------------------------------------------------------------- #
class CreateTask(BaseModel):
    prompt: str
    device_id: str = ""


class Approve(BaseModel):
    approved: bool
    note: str = ""


class Answer(BaseModel):
    facts: dict = {}


@app.get("/health")
async def health():
    from .vision.ocr import available as ocr_available
    return {"ok": True, "llm": app.state.llm.active, "ocr": ocr_available(),
            "db": "postgres" if getattr(app.state, "db", None) else "in-memory",
            "devices": list(app.state.devices._sessions.keys())}


@app.get("/skills")
async def skills():
    return [{"key": s.key, "title": s.title, "apps": s.app_packages,
             "approval_boundary": s.approval_boundary} for s in registry.all_skills()]


@app.post("/tasks")
async def create_task(body: CreateTask):
    task = Task(prompt=body.prompt, device_id=body.device_id)
    app.state.store.create(task)
    # Fire the driver in the background; the client polls or subscribes to events.
    asyncio.create_task(app.state.controller.run_task(task))
    return {"task_id": task.id, "state": task.state.value}


@app.get("/tasks")
async def list_tasks():
    return [_task_view(t) for t in app.state.store.all()]


@app.get("/tasks/{task_id}")
async def get_task(task_id: str):
    task = app.state.store.get(task_id)
    if not task:
        return {"error": "not found"}
    return _task_view(task, full=True)


@app.post("/tasks/{task_id}/approve")
async def approve(task_id: str, body: Approve):
    ok = app.state.controller.resolve_approval(task_id, body.approved, body.note)
    return {"ok": ok}


@app.post("/tasks/{task_id}/answer")
async def answer(task_id: str, body: Answer):
    ok = app.state.controller.resolve_answer(task_id, body.facts)
    return {"ok": ok}


@app.post("/tasks/{task_id}/cancel")
async def cancel(task_id: str):
    app.state.controller.cancel(task_id)
    return {"ok": True}


@app.post("/stop")
async def stop_all():
    """Emergency stop: cancel every task that isn't already finished."""
    from .agent.state import TERMINAL
    stopped = 0
    for t in app.state.store.all():
        if t.state not in TERMINAL:
            app.state.controller.cancel(t.id)
            stopped += 1
    return {"stopped": stopped}


@app.post("/voice/tts")
async def voice_tts(body: dict):
    """Text -> speech (WAV) via Groq PlayAI. 503 if unavailable (phone falls back)."""
    from .voice.groq_audio import tts
    audio = await tts(body.get("text", ""))
    if not audio:
        return Response(status_code=503)
    return Response(content=audio, media_type="audio/wav")


@app.post("/voice/stt")
async def voice_stt(file: UploadFile = File(...)):
    """Speech -> text via Groq Whisper."""
    from .voice.groq_audio import stt
    text = await stt(await file.read(), file.filename or "audio.wav")
    return {"text": text}


def _task_view(task: Task, full: bool = False) -> dict:
    view = {
        "task_id": task.id, "state": task.state.value, "prompt": task.prompt,
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
# WebSockets
# --------------------------------------------------------------------------- #
@app.websocket("/ws/device/{device_id}")
async def ws_device(socket: WebSocket, device_id: str):
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
            if mtype == "approval_response":
                controller.resolve_approval(msg["task_id"], bool(msg.get("approved")),
                                            msg.get("note", ""))
            elif mtype == "answer":
                controller.resolve_answer(msg["task_id"], msg.get("facts", {}))
            else:
                session.on_message(msg)   # correlate observe/act responses by rid
    except WebSocketDisconnect:
        pass
    finally:
        session.fail_all(RuntimeError("device disconnected"))
        app.state.devices.remove(device_id)


@app.websocket("/ws/events")
async def ws_events(socket: WebSocket):
    await socket.accept()
    q = app.state.store.subscribe()
    try:
        while True:
            event = await q.get()
            await socket.send_json(event)
    except WebSocketDisconnect:
        pass
    finally:
        app.state.store.unsubscribe(q)


if __name__ == "__main__":
    import uvicorn
    s = get_settings()
    uvicorn.run("app.main:app", host=s.host, port=s.port, reload=False)
