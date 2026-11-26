"""
Desktop executor: the Windows companion (desktop/onepiece_desktop) against a real WinForms
app, then a full E2E run — real uvicorn server, real WebSocket, paired desktop device,
the real ReAct graph + policy engine + approvals (decisions scripted, so no LLM).

The test app starts MINIMIZED and is driven only through UI Automation patterns, so the
tests never take focus or move the real mouse/keyboard.

Run:  python -m tests.test_desktop          (Windows only)
"""
from __future__ import annotations

import asyncio
import ctypes
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ["AUTH_SECRET"] = "test-secret-" + "d" * 40
os.environ["DATABASE_URL"] = ""
os.environ["REDIS_URL"] = ""
os.environ["EMBEDDING_BACKEND"] = "hashing"
os.environ["GROQ_API_KEY"] = ""
os.environ["GEMINI_API_KEY"] = ""
os.environ["MCP_CONFIG_PATH"] = "does-not-exist.json"
os.environ["MEMORY_LEARN"] = "false"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "desktop"))
TMP = Path(tempfile.mkdtemp(prefix="onepiece-desk-"))
os.environ["ONEPIECE_HOME"] = str(TMP)

from onepiece_desktop.agent import DesktopAgent          # noqa: E402
from onepiece_desktop.safety import Safety                # noqa: E402
from onepiece_desktop.uia import WindowsDesktop, capture_window  # noqa: E402

from app.agent.state import TaskState                     # noqa: E402
from app.mobile.schema import ActionType, AgentAction, ScreenObservation  # noqa: E402

TITLE = "OnePiece CineTest"
APP = ROOT / "desktop" / "tests" / "cinetest_app.ps1"
user32 = ctypes.windll.user32
PASS = 0


def check(cond: bool, label: str) -> None:
    global PASS
    assert cond, f"FAIL: {label}"
    PASS += 1
    print(f"PASS {label}")


def start_app() -> subprocess.Popen:
    # A leftover test window (crashed earlier run) would be found instead of the new one.
    for _ in range(40):
        stale = user32.FindWindowW(None, TITLE)
        if not stale:
            break
        user32.PostMessageW(stale, 0x0010, 0, 0)                # WM_CLOSE
        time.sleep(0.25)
    p = subprocess.Popen(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
                          "-File", str(APP)], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    for _ in range(80):
        if user32.FindWindowW(None, TITLE):
            time.sleep(0.5)
            return p
        time.sleep(0.25)
    p.kill()
    raise RuntimeError("test app did not start")


def app_has_focus() -> bool:
    """Did the TEST APP become the foreground window? (The user may switch between their
    own windows while the suite runs, so we don't compare the global foreground.)"""
    fg = user32.GetForegroundWindow()
    return bool(fg) and fg == user32.FindWindowW(None, TITLE)


def stop_app(p: subprocess.Popen) -> None:
    p.kill()
    p.wait(10)


def safety(**kw) -> Safety:
    d = Path(tempfile.mkdtemp(dir=TMP))
    base = dict(allowed_processes={"powershell.exe"}, workspaces=[d / "ws"],
                log_path=d / "actions.jsonl", stop_file=d / "STOP", cursor=lambda: (500, 500))
    base.update(kw)
    return Safety(**base)


def ref(screen: dict, label: str) -> int:
    return next(e["ref"] for e in screen["elements"] if label in (e["text"], e["desc"]))


def el(screen: dict, rid: str) -> dict:
    return next(e for e in screen["elements"] if e["resource_id"] == rid)


# --------------------------------------------------------------------------- #
# 1. Local safety controls (no UI)
# --------------------------------------------------------------------------- #
def unit_safety() -> None:
    s = safety()
    ws = s.workspaces[0]
    check(s.path_allowed("Scene") and s.path_allowed("2 tickets, 7:00 PM"),
          "plain text is not treated as a path")
    check(s.path_allowed(str(ws / "notes" / "a.txt")), "path inside the workspace allowed")
    check(not s.path_allowed(r"C:\Windows\System32\drivers\etc\hosts"),
          "path outside the workspace refused")
    check(not s.path_allowed(str(ws / ".." / ".." / "secret.txt")),
          "'..' traversal out of the workspace refused")
    check(not s.path_allowed("save to ~/.ssh/id_rsa") and not s.path_allowed(r"\\server\share\x"),
          "home and UNC paths refused")
    check(s.process_allowed("PowerShell.EXE") and not s.process_allowed("chrome.exe"),
          "process allowlist (case-insensitive)")

    pos = [(400, 300)]
    k = safety(cursor=lambda: pos[0])
    check(k.check_kill() == "", "kill switch idle")
    pos[0] = (0, 1)
    check("corner" in k.check_kill(), "mouse in the top-left corner engages the kill switch")
    pos[0] = (400, 300)
    check(k.check_kill() != "", "kill switch LATCHES after the mouse leaves the corner")
    k.reset_kill()
    check(k.check_kill() == "", "reset clears it")
    k.stop_file.parent.mkdir(parents=True, exist_ok=True)
    k.stop_file.touch()
    check("STOP" in k.check_kill(), "STOP file engages the kill switch")


# --------------------------------------------------------------------------- #
# 2. The UIA adapter against the real (minimized) app
# --------------------------------------------------------------------------- #
async def unit_adapter() -> None:
    p = start_app()
    s = safety()
    s.workspaces[0].mkdir(parents=True)
    d = WindowsDesktop(s, title_re=TITLE)
    try:
        o = await d.run(d.observe)
        scr = ScreenObservation(**o)                           # same contract as the phone
        check(o["app_package"] == "powershell.exe" and o["activity"] == TITLE,
              "observe: exe as app_package, window title as activity")
        roles = {e["role"] for e in o["elements"]}
        check({"Edit", "Button", "CheckBox", "Text"} <= roles and
              not any(e["text"] in ("Minimize", "Close") for e in o["elements"]),
              "UIA tree serialized (title-bar chrome skipped)")
        search = el(o, "searchBox")
        check(search["editable"] and search["desc"] == "Search movies" and not search["clickable"],
              "edit field: editable, label in desc")
        pin = el(o, "pinBox")
        check(pin["desc"] == "password field" and not pin["editable"] and pin["text"] == "",
              "password field marked, value never read, not offered as editable")
        check(len({e.ref for e in scr.elements}) == len(scr.elements), "refs are unique")

        r = await d.run(d.act, {"type": "type", "ref": search["ref"], "text": "Scene"})
        r2 = await d.run(d.act, {"type": "enter"})
        o = await d.run(d.observe)
        check(r["ok"] and r2["ok"] and "Results for Scene" in el(o, "resultLabel")["text"],
              "type (ValuePattern) + enter (posted keys) submits — window still minimized")
        await d.run(d.act, {"type": "tap", "ref": ref(o, "Seat A1")})
        o = await d.run(d.observe)
        check(el(o, "seatA1")["checked"] is True and el(o, "totalLabel")["text"] == "Total Rs 250",
              "tap a checkbox (TogglePattern) updates the app")
        res = await d.run(d.act, {"type": "type", "ref": el(o, "pinBox")["ref"], "text": "1234"})
        check(not res["ok"] and "secrets" in res["error"], "typing into a password field refused")
        res = await d.run(d.act, {"type": "type", "ref": search["ref"],
                                  "text": r"C:\Windows\win.ini"})
        check(not res["ok"] and "workspace" in res["error"], "typing a path outside the workspace refused")
        res = await d.run(d.act, {"type": "tap", "ref": 999})
        check(not res["ok"] and "observe again" in res["error"], "stale/unknown ref refused")
        res = await d.run(d.act, {"type": "back"})
        check(not res["ok"], "unsupported actions return an error (no silent no-op)")
        check(not app_has_focus() and
              user32.IsIconic(user32.FindWindowW(None, TITLE)),
              "no focus stolen: foreground unchanged, app still minimized")

        # Screenshot of ONE window: restore WITHOUT activating, capture, re-minimize.
        hwnd = user32.FindWindowW(None, TITLE)
        user32.ShowWindow(hwnd, 4)                              # SW_SHOWNOACTIVATE
        time.sleep(0.8)
        shot = await d.run(capture_window, hwnd)
        user32.ShowWindow(hwnd, 7)                              # SW_SHOWMINNOACTIVE
        import base64
        check(shot is not None and base64.b64decode(shot)[:2] == b"\xff\xd8",
              "PrintWindow screenshot of the target window only (JPEG)")
        check(not app_has_focus(), "...and still no focus change")

        # Allowlist: same window, but powershell.exe NOT allowed.
        blocked = WindowsDesktop(safety(allowed_processes={"notepad.exe"}), title_re=TITLE)
        o = await blocked.run(blocked.observe)
        check(o["activity"] == "BLOCKED" and len(o["elements"]) == 1 and not o["screenshot_b64"]
              and "allowlist" in o["elements"][0]["text"],
              "non-allowlisted window: no tree, no screenshot, just a notice")
        res = await blocked.run(blocked.act, {"type": "tap", "ref": 0})
        check(not res["ok"] and "allowlist" in res["error"], "acting on a non-allowlisted window refused")
        blocked.close()

        # open_app retargets to an open allowlisted window by title (no focus change).
        d2 = WindowsDesktop(s)
        res = await d2.run(d2.act, {"type": "open_app", "text": "cinetest"})
        o = await d2.run(d2.observe)
        check(res["ok"] and o["activity"] == TITLE, "open_app retargets to a window by title")
        res = await d2.run(d2.act, {"type": "open_app", "text": "regedit"})
        check(not res["ok"] and "not a configured app" in res["error"],
              "open_app refuses apps that aren't configured")
        d2.close()
    finally:
        d.close()
        stop_app(p)


# --------------------------------------------------------------------------- #
# 3. E2E: server + paired desktop over a real WebSocket + the real ReAct graph
# --------------------------------------------------------------------------- #
def free_port() -> int:
    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        return so.getsockname()[1]


async def e2e() -> None:
    import httpx
    import uvicorn
    from app.main import app

    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                           log_level="warning", lifespan="on"))
    srv_task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)
    base = f"http://127.0.0.1:{port}"
    p = start_app()
    agent = None
    agent_task = None
    try:
        auth = app.state.auth
        admin = {"Authorization": f"Bearer {auth.issue_admin()}"}
        async with httpx.AsyncClient(base_url=base, timeout=30) as http:
            code = (await http.post("/v1/devices/pair/start", headers=admin)).json()["code"]
            bad = await http.post("/v1/devices/pair", json={"code": code, "device_id": "browser-x"})
            check(bad.status_code == 400 and "reserved" in bad.text,
                  "pairing refuses reserved executor ids (browser-/api-)")
            r = await http.post("/v1/devices/pair", json={"code": code, "device_id": "desktop-test"})
            check(r.status_code == 200, "desktop pairs with a 6-digit code like the phone")
            token = r.json()["token"]

            s = safety()
            s.workspaces[0].mkdir(parents=True)
            desk = WindowsDesktop(s, title_re=TITLE, screenshots=False)
            pushes: list[dict] = []

            def user_pays() -> None:           # the HUMAN completes the payment step
                import uiautomation as auto
                with auto.UIAutomationInitializerInThread():
                    w = auto.WindowControl(searchDepth=1, Name=TITLE)
                    w.ButtonControl(AutomationId="payBtn").GetInvokePattern().Invoke()

            async def on_push(msg: dict) -> None:
                pushes.append(msg)
                if msg.get("type") == "approval_request":
                    # The person at the desktop approves on their console.
                    await agent.approve(msg["task_id"], True, "test user")
                elif msg.get("type") == "handoff":
                    await asyncio.get_running_loop().run_in_executor(None, user_pays)
                    agent.resume_file.touch()

            bad_agent = DesktopAgent(base, "desktop-test", "garbage", desk, s)
            await asyncio.wait_for(bad_agent.run_forever(), 15)
            check(bad_agent.auth_rejected, "bad token rejected at the WebSocket handshake")

            agent = DesktopAgent(base, "desktop-test", token, desk, s, on_push=on_push,
                                 handoff_timeout_s=60)
            agent_task = asyncio.create_task(agent.run_forever())
            await asyncio.wait_for(agent.connected.wait(), 15)
            for _ in range(50):
                if app.state.devices.get("desktop-test") is not None:
                    break
                await asyncio.sleep(0.1)
            sess = app.state.devices.get("desktop-test")
            check(sess is not None and sess.kind == "desktop", "server registers it as a desktop session")
            h = (await http.get("/v1/devices", headers=admin)).json()
            check("desktop-test" in str(h), "listed among devices")

            c = app.state.controller

            async def fake_plan(task):
                task.goal, task.skill = "book 2 seats for Scene in CineTest", "generic"
                task.plan = ["search", "pick seats", "stop at pay"]
                task.set_state(TaskState.PLANNING)
            c._plan_phase = fake_plan

            seen: list[str] = []

            async def decide(task, skill, screen, feedback=""):
                o = screen.model_dump()
                seen.append(screen.activity)
                if el(o, "statusLabel")["text"].endswith("PAYMENT STARTED"):
                    return AgentAction(type=ActionType.DONE, text="Seats A1+A2 booked; paid by user")
                if el(o, "resultLabel")["text"] == "No results":
                    if not any(x.startswith("type") for x in task.history):
                        return AgentAction(type=ActionType.TYPE, ref=el(o, "searchBox")["ref"],
                                           text="Scene")
                    return AgentAction(type=ActionType.ENTER)
                if not any("REFUSED" in x for x in task.history):
                    # One refused action: the model must SEE why (history), not loop on it.
                    return AgentAction(type=ActionType.TYPE, ref=el(o, "searchBox")["ref"],
                                       text=r"C:\Windows\win.ini", thought="workspace probe")
                for rid in ("seatA1", "seatA2"):
                    if not el(o, rid)["checked"]:
                        return AgentAction(type=ActionType.TAP, ref=el(o, rid)["ref"])
                return AgentAction(type=ActionType.TAP, ref=el(o, "payBtn")["ref"])
            c._decide = decide

            r = await http.post("/v1/tasks", headers=admin, json={
                "prompt": "On my desktop, book Scene in the CineTest app: 2 seats"})
            tid = r.json()["task_id"]
            task = app.state.store.get(tid)
            for _ in range(600):
                if task.state in (TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED,
                                  TaskState.WAITING_FOR_USER) and not task.pending_approval_id:
                    break
                await asyncio.sleep(0.1)
            hist = task.history
            print("    history:", hist)
            print("    state:", task.state.value, "| pushes:", [m.get("type") for m in pushes])
            for e in task.events[-8:]:
                print("     ", e.kind, e.detail[:140])
            check(task.executor == "desktop" and task.device_id == "desktop-test",
                  "auto-selected the desktop executor ('on my desktop') and bound the device")
            check(not any(x.startswith("open_app") for x in hist),
                  "no phone-style app bootstrap on the desktop")
            check(any(x.startswith("type 'Scene'") for x in hist) and "enter" in hist,
                  "agent typed + submitted in the desktop app over the WebSocket")
            check(any("REFUSED" in x and "workspace" in x for x in hist),
                  "local workspace refusal fed back into the model's history")
            recs = c.approvals_for(tid)
            check(recs and recs[-1].risk_level == 4 and recs[-1].decision == "approved",
                  "'Proceed to pay' needed a Level-4 approval (approved from the desktop)")
            check(any(m.get("type") == "handoff" for m in pushes),
                  "payment handed to the user at the desktop (handoff)")
            check(any(a.endswith("PAYMENT STARTED") for a in [el(ScreenObservation(**(
                      await desk.run(desk.observe))).model_dump(), "statusLabel")["text"]]) and
                  task.state in (TaskState.COMPLETED, TaskState.WAITING_FOR_USER) and
                  "done" in " ".join(e.detail for e in task.events),
                  f"resumed after the user paid, then reported done ({task.state.value})")
            log_lines = s.log_path.read_text(encoding="utf-8").splitlines()
            check(any('"handoff"' in x for x in log_lines) and
                  any('"type": "tap"' in x for x in log_lines),
                  "every action + the handoff written to the local action log")

            # ---- kill switch: mouse into the corner mid-task -> task cancelled ----
            stop_app(p)                                  # fresh app state
            p = start_app()
            desk.target_hwnd = 0
            s.cursor = lambda: (0, 0)
            r = await http.post("/v1/tasks", headers=admin, json={
                "prompt": "On my desktop, book Scene in the CineTest app: 2 seats",
                "executor": "desktop"})
            task2 = app.state.store.get(r.json()["task_id"])
            for _ in range(300):
                if task2.state in (TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED):
                    break
                await asyncio.sleep(0.1)
            o = await desk.run(desk.observe)
            check(task2.state == TaskState.CANCELLED and
                  any("kill switch" in e.detail for e in task2.events),
                  "kill switch: device refuses + backend cancels the task")
            check(not el(o, "seatA1")["checked"], "...and nothing was changed on the desktop")
            check(not app_has_focus(), "E2E never took focus")
    finally:
        if agent is not None:
            await agent.close()
        if agent_task is not None:
            agent_task.cancel()
        stop_app(p)
        server.should_exit = True
        await srv_task


async def main() -> None:
    unit_safety()
    await unit_adapter()
    await e2e()
    print(f"\nDESKTOP PASS  ({PASS} checks)")


if __name__ == "__main__":
    asyncio.run(main())
