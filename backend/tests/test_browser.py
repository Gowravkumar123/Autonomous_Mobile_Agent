"""
Browser executor: DOM serialization, actions, safety controls, and a full ReAct run on a
local test site through the real graph + policy engine (decisions scripted, so no LLM).

Run:  python -m tests.test_browser
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "")
os.environ.setdefault("GROQ_API_KEY", "")

from app.agent.controller import AgentController          # noqa: E402
from app.agent.memory import MemoryManager                # noqa: E402
from app.agent.state import Task, TaskState               # noqa: E402
from app.config import Settings                           # noqa: E402
from app.executors.browser import BrowserSession, domain_allowed  # noqa: E402
from app.llm.router import LLMRouter                      # noqa: E402
from app.mobile.schema import ActionType, AgentAction, ScreenObservation  # noqa: E402
from app.policy.engine import PolicyEngine                # noqa: E402
from app.store import TaskStore                           # noqa: E402
from app.ws import DeviceRegistry                         # noqa: E402
from tests.site_server import serve                       # noqa: E402

PASS = 0


def check(cond: bool, label: str) -> None:
    global PASS
    assert cond, f"FAIL: {label}"
    PASS += 1
    print(f"PASS {label}")


def ref_of(screen: ScreenObservation, text: str, role: str = "") -> int | None:
    t = text.lower()
    for e in screen.elements:
        if (e.text.strip().lower() == t or e.desc.strip().lower() == t) and \
                (not role or e.role.startswith(role)):
            return e.ref
    return None


async def unit(base: str):
    s = await BrowserSession("browser-unit", allowed_domains=("127.0.0.1",)).start(
        base + "/index.html")
    try:
        scr = ScreenObservation(**(await s.request("observe"))["screen"])
        search = next(e for e in scr.elements if e.desc == "Search movies")
        check(scr.app_package == "127.0.0.1" and "CineTest" in scr.activity,
              "observe: host as app_package, title|url as activity")
        check(search.editable and not search.clickable, "search input serialized as editable")
        check(len({e.ref for e in scr.elements}) == len(scr.elements), "refs are unique")
        check(bool(scr.screenshot_b64), "observation carries a screenshot (for vision)")

        await s.request("act", action={"type": "type", "ref": search.ref, "text": "Scene"})
        await s.request("act", action={"type": "enter"})
        scr = ScreenObservation(**(await s.request("observe"))["screen"])
        check("results.html?q=Scene" in scr.activity and ref_of(scr, "Book tickets") is not None,
              "type + enter submits the form")
        await s.request("act", action={"type": "back"})
        scr = ScreenObservation(**(await s.request("observe"))["screen"])
        check("index.html" in scr.activity, "back navigates back")

        res = await s.request("act", action={"type": "open_app", "text": "example.com"})
        scr = ScreenObservation(**(await s.request("observe"))["screen"])
        check(not res["ok"] and "allowlist" in res["error"] and "127.0.0.1" == scr.app_package,
              "domain allowlist blocks off-site navigation and stays on the page")
        await s.request("act", action={"type": "tap", "ref": ref_of(scr, "Partner site")})
        scr = ScreenObservation(**(await s.request("observe"))["screen"])
        check("index.html" in scr.activity, "tapping an off-site link is blocked too (stays put)")

        await s.request("act", action={"type": "tap", "ref": ref_of(scr, "Quick book")})
        scr = ScreenObservation(**(await s.request("observe"))["screen"])
        check(ref_of(scr, "DISMISSED") is not None and ref_of(scr, "CONFIRMED") is None,
              "confirm() dialogs are dismissed, never auto-accepted")

        check(s._to_url("bookmyshow.com") == "https://bookmyshow.com" and
              "duckduckgo" in s._to_url("scene movie showtimes"),
              "open_app: domain -> https URL, words -> web search")
        check((await s.request("handoff"))["stop"], "handoff stops (headless can't pay for you)")
        png = await s.evidence()
        check(png is not None and png[:4] == b"\x89PNG", "evidence: full-page PNG screenshot")
    finally:
        await s.close()
    check(domain_allowed("https://m.bookmyshow.com/x", ("bookmyshow.com",)) and
          not domain_allowed("https://evilbookmyshow.com", ("bookmyshow.com",)),
          "allowlist matches subdomains but not look-alike domains")


async def graph_run(base: str):
    store, devices = TaskStore(), DeviceRegistry()
    c = AgentController(LLMRouter(Settings(groq_api_key="", gemini_api_key="")),
                        PolicyEngine(spend_threshold=0), devices, store, MemoryManager())

    async def decide(task, skill, screen, feedback=""):
        a = screen.activity
        if "index.html" in a:
            if not any(h.startswith("type") for h in task.history):
                return AgentAction(type=ActionType.TYPE, ref=ref_of(screen, "Search movies"),
                                   text="Scene")
            return AgentAction(type=ActionType.ENTER)
        if "results.html" in a:
            return AgentAction(type=ActionType.SCROLL, direction="down")   # CTA override test
        if "show.html" in a:
            return AgentAction(type=ActionType.TAP, ref=ref_of(screen, "7:00 PM"))
        if "seats.html" in a:
            for seat in ("Seat A1", "Seat A2"):
                el = next(e for e in screen.elements if e.desc == seat)
                if not el.checked:
                    return AgentAction(type=ActionType.TAP, ref=el.ref)
            return AgentAction(type=ActionType.TAP, ref=ref_of(screen, "Proceed to pay", "button"))
        return AgentAction(type=ActionType.FAIL, text=f"unexpected page {a}")
    c._decide = decide                                   # type: ignore[assignment]

    task = Task(prompt="book Scene on the website, 2 seats", owner="admin",
                start_url=base + "/index.html", goal="book 2 seats for Scene",
                skill="generic", plan=["search", "pick show", "pick seats", "stop at pay"])
    store.create(task)
    check(c._choose_executor(task) == "browser", "select_executor picks browser (start_url)")

    async def approve_when_asked():
        for _ in range(400):
            if task.pending_approval_id:
                c.decide_approval(task.pending_approval_id, True, approver="admin")
                return
            await asyncio.sleep(0.05)
    watcher = asyncio.create_task(approve_when_asked())

    session = await c._select_executor(task)
    check(session is not None and task.device_id == f"browser-{task.id}" and
          devices.get(task.device_id) is session, "browser session registered like a device")
    try:
        await asyncio.wait_for(c._execute_phase(task), 120)
    finally:
        await c._close_browser(task, session)
    watcher.cancel()

    h = task.history
    print("    history:", h)
    check(any(x.startswith("type 'Scene'") for x in h) and "enter" in h,
          "agent typed the query and submitted")
    check(any("override scroll" in e.detail and "Book tickets" in e.detail for e in task.events),
          "CTA override tapped 'Book tickets' (not the 'Facebook' link)")
    check(sum(1 for x in h if x.startswith("tap #")) >= 3, "picked showtime + 2 seats (auto-allowed)")
    recs = c.approvals_for(task.id)
    check(recs and recs[0].risk_level == 4 and recs[0].decision == "approved",
          "'Proceed to pay' required a Level-4 approval")
    check(task.state == TaskState.WAITING_FOR_USER and "seats.html" in task.summary,
          "payment handed to the user with the URL to finish at")
    check(task.evidence and (Path(__file__).resolve().parents[1] / "evidence" /
                             task.evidence[0]).is_file(), "final screenshot saved as evidence")
    check(devices.get(f"browser-{task.id}") is None, "browser session closed + unregistered")


async def main():
    srv, base = serve()
    try:
        await unit(base)
        await graph_run(base)
    finally:
        srv.shutdown()
    print(f"\nBROWSER PASS  ({PASS} checks)")


if __name__ == "__main__":
    asyncio.run(main())
