"""
Transaction-bound approvals (blueprint section 7, "Approval protocol").

An approval is not a bare yes/no on a task: it is a RECORD bound to the exact action the
agent proposed — the action type/target, the label of the element it would tap, and the
amount on screen — with a short expiry. Right before executing, the agent re-reads the
screen and recomputes the binding; if anything material changed (different button under
that ref, different amount), the approval is invalidated and must be requested again.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field

from ..mobile.schema import AgentAction, ScreenObservation

PENDING, APPROVED, REJECTED, EXPIRED, INVALIDATED = (
    "pending", "approved", "rejected", "expired", "invalidated")


def action_binding(action: AgentAction, screen: ScreenObservation) -> str:
    """Stable hash of what the user is actually approving."""
    from ..policy.engine import _detect_amount, _element_label, _screen_text
    material = {
        "type": action.type.value if hasattr(action.type, "value") else str(action.type),
        "ref": action.ref,
        "text": action.text or "",
        "x": round(action.x, 3) if action.x is not None else None,
        "y": round(action.y, 3) if action.y is not None else None,
        "target": _element_label(action.ref, screen).strip(),
        "amount": _detect_amount(_screen_text(screen)),
        "app": screen.app_package,
        # Tool calls: the approval covers this exact tool with these exact arguments.
        "tool": getattr(action, "tool", None) or "",
        "args": getattr(action, "args", None) or {},
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True, default=str).encode()).hexdigest()


@dataclass
class ApprovalRecord:
    task_id: str
    step_id: int
    action_hash: str
    expires_at: float
    title: str = ""
    reason: str = ""
    amount: float | None = None
    action_label: str = ""
    risk_level: int = 3
    approval_id: str = field(default_factory=lambda: "apr_" + uuid.uuid4().hex[:16])
    created_at: float = field(default_factory=time.time)
    decision: str = PENDING
    approver: str = ""
    note: str = ""
    decided_at: float | None = None

    @property
    def expired(self) -> bool:
        return self.decision == PENDING and time.time() >= self.expires_at

    def decide(self, decision: str, approver: str = "", note: str = "") -> None:
        self.decision = decision
        self.approver = approver
        self.note = note
        self.decided_at = time.time()

    def view(self) -> dict:
        d = asdict(self)
        d["expires_in"] = max(0, int(self.expires_at - time.time())) if self.decision == PENDING else 0
        return d
