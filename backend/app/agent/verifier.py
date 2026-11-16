"""
Verification engine. "A payment screen is not proof of payment." We only report
success when a trusted success marker is actually visible on screen.
"""
from __future__ import annotations

from ..mobile.schema import ScreenObservation
from ..skills.base import Skill


def screen_text(screen: ScreenObservation) -> str:
    return " ".join(f"{e.text} {e.desc}" for e in screen.elements).lower()


def is_success(screen: ScreenObservation, skill: Skill) -> bool:
    if not skill.success_markers:
        return False
    text = screen_text(screen)
    return any(marker in text for marker in skill.success_markers)
