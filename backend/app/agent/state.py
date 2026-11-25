"""Explicit task state so the agent can pause safely and resume later."""
from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from ..mobile.schema import AgentAction


class TaskState(str, Enum):
    CREATED = "CREATED"
    PLANNING = "PLANNING"
    RUNNING = "RUNNING"
    WAITING_FOR_USER = "WAITING_FOR_USER"
    PAUSED = "PAUSED"            # user paused; resumes at the next ReAct cycle boundary
    VERIFYING = "VERIFYING"
    RECOVERING = "RECOVERING"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


TERMINAL = {TaskState.COMPLETED, TaskState.CANCELLED, TaskState.FAILED}


class TaskEvent(BaseModel):
    ts: float = Field(default_factory=time.time)
    kind: str                      # state_change | action | observation | approval | note | error
    detail: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class Task(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    device_id: str = ""            # execution target (a phone, or "browser-<task id>")
    owner: str = ""                # who may see/control it: "admin" or a paired device id
    executor: str = ""             # "mobile" | "browser" ("" = auto-select after planning)
    start_url: str = ""            # optional starting page for the browser executor
    evidence: list[str] = Field(default_factory=list)   # e.g. final screenshot paths
    executor_hint: str = ""        # intent's suggestion: "api" if tools alone can do it
    # Normalized results of tool calls, newest last: {tool, args, ok, text, cached}.
    tool_results: list[dict] = Field(default_factory=list)
    prompt: str = ""
    state: TaskState = TaskState.CREATED

    # Filled during planning.
    goal: str = ""
    skill: str = "generic"
    app_hint: str = ""
    facts: dict[str, Any] = Field(default_factory=dict)
    plan: list[str] = Field(default_factory=list)
    approval_boundary: str = ""
    clarifications: list[str] = Field(default_factory=list)
    # Parallel to `clarifications`: 2-5 concrete choices per question (may be empty).
    clarification_choices: list[list[str]] = Field(default_factory=list)
    # The user pre-authorized commitments ("do it", "send it"): skip Level-3 approval
    # pauses. Level 4 (payment / OTP / biometric) ALWAYS pauses regardless.
    pre_authorized: bool = False

    # Runtime.
    step_id: int = 0
    approvals_granted: int = 0                          # how many sensitive steps the user approved
    history: list[str] = Field(default_factory=list)   # short human-readable action log
    pending_action: Optional[AgentAction] = None       # action awaiting approval
    pending_approval_reason: str = ""
    summary: str = ""
    error: str = ""
    cancel_requested: bool = False
    pause_requested: bool = False
    # The approval currently awaiting a decision (transaction-bound; see controller).
    pending_approval_id: str = ""

    events: list[TaskEvent] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    def log(self, kind: str, detail: str = "", **data: Any) -> None:
        self.events.append(TaskEvent(kind=kind, detail=detail, data=data))
        self.updated_at = time.time()

    def set_state(self, state: TaskState, detail: str = "") -> None:
        self.state = state
        self.log("state_change", detail or state.value, state=state.value)
