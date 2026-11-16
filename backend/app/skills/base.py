"""
A Skill is a *playbook*, not a hard-coded script.

The generic observe->plan->act->verify loop already works on any app. A skill just
injects domain knowledge — which app, the usual steps, what fields matter, where the
approval boundary is, and how to tell success — so common tasks run more reliably.
Adding a skill never touches the agent core.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Skill:
    key: str
    title: str
    app_packages: list[str] = field(default_factory=list)
    # Free-text guidance appended to the planner/action prompts.
    playbook: str = ""
    # Where to stop and hand control to the user.
    approval_boundary: str = ""
    # Substrings that, if visible on screen, indicate the goal is complete.
    success_markers: list[str] = field(default_factory=list)
    # Required facts; if missing, the agent should ask before starting.
    required_facts: list[str] = field(default_factory=list)

    def primary_app(self) -> str:
        return self.app_packages[0] if self.app_packages else ""
