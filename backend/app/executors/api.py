"""
API/MCP executor (blueprint select_executor: "browser, mobile, desktop or API/MCP").

For goals that need no UI at all (look something up, call a service), the task runs
against the tool registry only. `ToolSession` speaks the same device contract as the
phone and the browser so the ReAct graph is unchanged: it "observes" an empty workspace
(the tool results are fed to the model separately) and refuses UI actions.
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable


class ToolSession:
    kind = "api"

    def __init__(self, device_id: str,
                 notify: Callable[[str, dict], Awaitable[None]] | None = None):
        self.device_id = device_id
        self.notify = notify

    async def request(self, type_: str, timeout: float = 60.0, **data: Any) -> dict:
        if type_ == "observe":
            return {"screen": {"app_package": "tools", "activity": "API/MCP workspace",
                               "elements": [], "screenshot_b64": None}}
        if type_ == "act":
            return {"ok": False,
                    "error": "UI actions are unavailable in the API executor — use call_tool"}
        if type_ == "handoff":
            return {"ok": True, "stop": True,
                    "message": "This step needs you — finish it yourself, then tell me."}
        return {}

    async def push(self, type_: str, **data: Any) -> None:
        if self.notify is not None:
            try:
                await self.notify(type_, data)
            except Exception:
                pass

    def on_message(self, msg: dict) -> None:
        pass

    def fail_all(self, exc: Exception) -> None:
        pass
