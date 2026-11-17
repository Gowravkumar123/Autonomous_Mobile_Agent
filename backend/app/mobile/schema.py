"""
The device <-> brain contract.

The phone sends OBSERVATIONS (a serialized accessibility tree). The brain replies
with a single ACTION. Everything is plain JSON so the Kotlin client and the Python
brain stay decoupled. This is the most important boundary in the system — keep it
stable and small.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------- #
# Screen observation (phone -> brain)
# --------------------------------------------------------------------------- #
class UiElement(BaseModel):
    """One node from the Android accessibility tree.

    We identify controls by stable, semantic properties (text / id / role),
    NOT by raw pixel coordinates. `bounds` is kept only as a fallback the
    phone uses to actually tap — the model should target by `ref`.
    """
    ref: int = Field(..., description="Stable index assigned by the serializer this frame.")
    depth: int = Field(0, description="Nesting level in the UI tree (0 = top).")
    parent: int = Field(-1, description="ref of the nearest kept ancestor (-1 = root); groups elements.")
    role: str = Field("", description="e.g. Button, EditText, TextView, CheckBox.")
    text: str = Field("", description="Visible text.")
    desc: str = Field("", description="contentDescription.")
    resource_id: str = Field("", description="Android view id, e.g. com.app:id/search.")
    clickable: bool = False
    editable: bool = False
    scrollable: bool = False
    checked: Optional[bool] = None
    enabled: bool = True
    bounds: list[int] = Field(default_factory=list, description="[left, top, right, bottom] px.")

    def summary(self) -> str:
        """Compact one-line form the LLM reads. Pixel bounds are omitted on purpose."""
        tags = []
        if self.clickable:
            tags.append("clickable")
        if self.editable:
            tags.append("editable")
        if self.scrollable:
            tags.append("scrollable")
        if self.checked is not None:
            tags.append(f"checked={self.checked}")
        if not self.enabled:
            tags.append("disabled")
        label = self.text or self.desc or self.resource_id or self.role or "?"
        extra = f" [{', '.join(tags)}]" if tags else ""
        rid = f" id={self.resource_id}" if self.resource_id else ""
        return f"#{self.ref} {self.role}: {label!r}{rid}{extra}"


class ScreenObservation(BaseModel):
    """A full snapshot of what is currently on screen."""
    app_package: str = ""
    activity: str = ""
    elements: list[UiElement] = Field(default_factory=list)
    # Optional low-res screenshot (base64) for the vision fallback path.
    screenshot_b64: Optional[str] = None
    is_keyboard_open: bool = False

    def render_for_llm(self, limit: int = 120) -> str:
        head = f"App: {self.app_package or 'unknown'}  Screen: {self.activity or '?'}"
        lines = [el.summary() for el in self.elements[:limit]]
        more = "" if len(self.elements) <= limit else f"\n… (+{len(self.elements) - limit} more elements)"
        return head + "\n" + "\n".join(lines) + more


# --------------------------------------------------------------------------- #
# Actions (brain -> phone)
# --------------------------------------------------------------------------- #
class ActionType(str, Enum):
    TAP = "tap"
    TAP_XY = "tap_xy"      # tap a point (normalized x/y) when no element ref is available
    TYPE = "type"
    ENTER = "enter"        # press the keyboard Go/Search/Enter to submit the current field
    SCROLL = "scroll"
    SWIPE = "swipe"
    BACK = "back"
    HOME = "home"
    OPEN_APP = "open_app"
    WAIT = "wait"
    # Terminal / control signals (handled by the brain, surfaced to the phone/UI):
    ASK_USER = "ask_user"        # need a clarification before proceeding
    REQUEST_APPROVAL = "approve" # sensitive step: hand control to the user
    DONE = "done"                # task goal verified complete
    FAIL = "fail"                # cannot safely proceed


class AgentAction(BaseModel):
    """A single decision from the brain. Exactly one action per step."""
    type: ActionType
    ref: Optional[int] = Field(None, description="Target element ref (for tap/type/scroll).")
    text: Optional[str] = Field(None, description="Text to type, app to open, or message to the user.")
    direction: Optional[Literal["up", "down", "left", "right"]] = None
    seconds: Optional[float] = Field(None, description="For WAIT.")
    x: Optional[float] = Field(None, description="For tap_xy: horizontal position, 0.0-1.0 of screen width.")
    y: Optional[float] = Field(None, description="For tap_xy: vertical position, 0.0-1.0 of screen height.")
    # The model's own narration — shown in the UI log and used for verification.
    thought: str = ""
    expectation: str = Field("", description="What the screen should look like if this worked.")
    # For ASK_USER: concrete choices to offer the user (so the phone can read/show them).
    options: list[str] = Field(default_factory=list,
                               description="2-5 short choices to present when type=ask_user.")

    # Risk metadata filled in by the policy engine (NOT trusted from the LLM).
    risk_level: int = 0
    risk_reason: str = ""

    @field_validator("options", mode="before")
    @classmethod
    def _coerce_options(cls, v: Any) -> list:
        # The model emits "options": null for non-ask actions — treat as empty, and be
        # lenient if it returns a single string instead of a list.
        if v is None:
            return []
        if isinstance(v, str):
            return [v] if v.strip() else []
        return v


class Observation(BaseModel):
    """Envelope a device sends up the WebSocket."""
    task_id: str
    step_id: int
    screen: ScreenObservation


class ApprovalRequest(BaseModel):
    task_id: str
    step_id: int
    title: str
    summary: str
    amount: Optional[float] = None
    reason: str = ""


class ApprovalResponse(BaseModel):
    task_id: str
    step_id: int
    approved: bool
    note: str = ""
