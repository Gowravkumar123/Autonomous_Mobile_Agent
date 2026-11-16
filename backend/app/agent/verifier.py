"""
Verification engine. "A payment screen is not proof of payment." We only report
success when a trusted success marker is actually visible on screen.
"""
from __future__ import annotations

from ..mobile.schema import ScreenObservation
from ..skills.base import Skill


def screen_text(screen: ScreenObservation) -> str:
    return " ".join(f"{e.text} {e.desc}" for e in screen.elements).lower()


def is_success(screen: ScreenObservation, skill: Skill, task) -> bool:
    """
    True only when the goal is genuinely done. Guards against false positives from
    ambient UI text (the earlier bug: WhatsApp's chat list already says "delivered",
    so the task "completed" after merely opening the app).
    """
    # 1. The agent must have actually done meaningful work first.
    if len(task.history) < skill.min_actions_before_success:
        return False
    # 2. A commitment-type goal (e.g. sending) is only done once it was approved.
    if skill.confirm_after_approval and task.approvals_granted < 1:
        return False

    text = screen_text(screen)

    # 3. The task's key content (e.g. the exact message text) must be visible.
    if skill.success_requires_fact:
        val = str(task.facts.get(skill.success_requires_fact, "")).strip().lower()
        if not val or val not in text:
            return False

    # 4. A success marker must be present, if the skill defines any.
    if skill.success_markers:
        return any(marker in text for marker in skill.success_markers)

    # No markers, but the content/approval gates above passed.
    return bool(skill.success_requires_fact)
