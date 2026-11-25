"""Common tool contract (blueprint: "one action contract" — tools return normalized,
verifiable results regardless of where they run: built-in Python or an MCP server)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

ALLOW, APPROVE, DENY = "allow", "approve", "deny"


class ToolError(Exception):
    pass


@dataclass
class ToolResult:
    ok: bool
    text: str = ""                     # normalized, truncated text for the model
    structured: Any = None             # structured content when the tool provides it
    error: str = ""
    elapsed_s: float = 0.0
    cached: bool = False               # idempotency: returned the earlier result

    def for_model(self, limit: int = 1500) -> str:
        body = self.text if self.ok else f"ERROR: {self.error or self.text}"
        return body if len(body) <= limit else body[:limit] + " …(truncated)"


@dataclass
class ToolSpec:
    name: str                          # namespaced: "<server>.<tool>" / "web.search"
    description: str
    input_schema: dict
    source: str                        # "builtin" | "mcp:<server>"
    read_only: bool = False
    destructive: bool = False
    idempotent: bool = False
    policy: str = APPROVE              # allow | approve | deny (after config + annotations)
    risk: int = 3                      # 1 read-only auto, 3 needs approval, 4 financial
    remote_name: str = ""              # name on the MCP server
    handler: Callable[[dict], Awaitable[ToolResult]] | None = None   # built-ins
    meta: dict = field(default_factory=dict)

    def brief(self) -> dict:
        return {"name": self.name, "description": self.description[:300],
                "source": self.source, "read_only": self.read_only,
                "destructive": self.destructive, "policy": self.policy, "risk": self.risk,
                "input_schema": self.input_schema}
