"""
Unified tool registry (blueprint: "Tool execution layer — tool discovery, schema
validation, dispatch, runtime isolation and result normalization"; API/MCP adapter
controls: "schema validation, OAuth scopes, secret isolation, rate limits, idempotency").

Sources: built-in research tools + every MCP server in the config file. Each tool gets a
POLICY the agent cannot override:
  allow   -> runs automatically (read-only tools you trust)
  approve -> transaction-bound user approval first (default for MCP tools)
  deny    -> never runs
Resolution per MCP tool, first match wins:
  1. config  servers.<s>.tools.<tool> = allow|approve|deny
  2. config  servers.<s>.default_policy
  3. if servers.<s>.trust_annotations: read-only & non-destructive -> allow, else approve
  4. approve   (MCP annotations are untrusted hints per the spec, so we don't rely on them)
Tools whose name looks financial (pay/purchase/transfer/order/...) are risk 4: approval
that can never be pre-authorized. Blanket rules (default_policy, trust_annotations) can't
auto-allow them; only an EXPLICIT per-tool "allow" written by the owner can (progressive
autonomy: "expand only through explicit, scoped permissions").

Config file (JSON, gitignored — may reference secrets as ${ENV_VAR}, resolved from env):
{
  "servers": {
    "notes":   {"command": "python", "args": ["-m", "my_server"], "env": {"KEY": "${NOTES_KEY}"},
                "trust_annotations": true, "tools": {"delete_all": "deny"}},
    "remote":  {"url": "https://host/mcp", "headers": {"Authorization": "Bearer ${REMOTE_TOKEN}"},
                "default_policy": "approve"}
  }
}
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from .base import ALLOW, APPROVE, DENY, ToolError, ToolResult, ToolSpec

log = logging.getLogger("agent.tools")

_FINANCIAL = re.compile(r"pay|purchase|buy|checkout|transfer|withdraw|refund|charge|order|"
                        r"invoice|wallet|upi|card", re.I)
_ENV = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _subst_env(value: Any) -> Any:
    """Resolve ${ENV_VAR} references so secrets stay in .env, never in the config file."""
    if isinstance(value, str):
        return _ENV.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _subst_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_subst_env(v) for v in value]
    return value


def _norm_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]", "_", s)[:48]


class ToolRegistry:
    def __init__(self, config: dict | None = None, *, builtins: bool = True,
                 timeout_s: float = 30.0, rate_per_minute: int = 30,
                 max_calls_per_task: int = 25, allow_private_fetch: bool = False):
        self.config = config or {}
        self.use_builtins = builtins
        self.timeout_s = timeout_s
        self.rate_per_minute = rate_per_minute
        self.max_calls_per_task = max_calls_per_task
        self.allow_private_fetch = allow_private_fetch
        from ..kv import MemoryKV
        self.kv = MemoryKV()                           # shared via Redis when configured
        self._tools: dict[str, ToolSpec] = {}
        self._conns: dict[str, Any] = {}               # server name -> MCPConnection
        self._recent: dict[str, deque] = defaultdict(deque)       # rate limiting
        self._task_calls: dict[str, int] = defaultdict(int)       # per-task call budget
        self._idem: dict[str, ToolResult] = {}                    # task|tool|args -> result

    # ------------------------------------------------------------------ #
    # construction / lifecycle
    # ------------------------------------------------------------------ #
    @classmethod
    def from_file(cls, path: str | os.PathLike, **kw) -> "ToolRegistry":
        p = Path(path)
        cfg: dict = {}
        if p.is_file():
            try:
                cfg = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                log.error("invalid MCP config %s: %s", p, e)
        return cls(_subst_env(cfg), **kw)

    async def start(self) -> None:
        if self.use_builtins:
            from .builtins import builtin_specs
            for spec in builtin_specs(self.allow_private_fetch):
                self._tools[spec.name] = spec
        await self._connect_all()

    async def stop(self) -> None:
        for conn in self._conns.values():
            await conn.stop()
        self._conns.clear()

    async def refresh(self) -> dict:
        await self.stop()
        self._tools = {k: v for k, v in self._tools.items() if v.source == "builtin"}
        await self._connect_all()
        return self.status()

    async def _connect_all(self) -> None:
        from .mcp_client import MCPConnection
        servers = (self.config.get("servers") or self.config.get("mcpServers") or {})
        conns = {name: MCPConnection(_norm_name(name), cfg, self.timeout_s)
                 for name, cfg in servers.items() if not cfg.get("disabled")}
        await asyncio.gather(*(c.start() for c in conns.values()))
        for name, conn in conns.items():
            self._conns[name] = conn
            for t in conn.tools:
                spec = self._spec_from_mcp(name, servers[name], t)
                self._tools[spec.name] = spec

    def _spec_from_mcp(self, server: str, cfg: dict, tool) -> ToolSpec:
        ann = getattr(tool, "annotations", None)
        ro = bool(getattr(ann, "read_only_hint", False)) if ann else False
        destructive = bool(getattr(ann, "destructive_hint", True)) if ann else True
        idem = bool(getattr(ann, "idempotent_hint", False)) if ann else False
        overrides = cfg.get("tools") or {}
        explicit = tool.name in overrides
        if explicit:
            policy = overrides[tool.name]
        elif cfg.get("default_policy"):
            policy = cfg["default_policy"]
        elif cfg.get("trust_annotations"):
            policy = ALLOW if (ro and not destructive) else APPROVE
        else:
            policy = APPROVE
        policy = policy if policy in (ALLOW, APPROVE, DENY) else APPROVE
        financial = bool(_FINANCIAL.search(tool.name))
        if financial and policy == ALLOW and not explicit:
            policy = APPROVE        # blanket rules never auto-allow money-looking tools
        risk = 4 if (financial and policy != ALLOW) else (1 if policy == ALLOW else 3)
        return ToolSpec(
            name=f"{_norm_name(server)}.{tool.name}",
            description=(tool.description or getattr(tool, "title", "") or "").strip(),
            input_schema=dict(getattr(tool, "input_schema", None) or {"type": "object"}),
            source=f"mcp:{server}", read_only=ro, destructive=destructive, idempotent=idem,
            policy=policy, risk=risk, remote_name=tool.name)

    # ------------------------------------------------------------------ #
    # discovery
    # ------------------------------------------------------------------ #
    def list(self) -> list[ToolSpec]:
        return sorted(self._tools.values(), key=lambda t: t.name)

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return [t.name for t in self.list() if t.policy != DENY]

    def status(self) -> dict:
        return {"tools": len(self._tools),
                "servers": {n: {"connected": c.connected, "tools": len(c.tools),
                                "error": c.error} for n, c in self._conns.items()}}

    def catalog(self, max_tools: int = 40) -> str:
        """Compact catalog for the prompt: name, policy, description, argument schema."""
        lines = []
        for t in self.list()[:max_tools]:
            if t.policy == DENY:
                continue
            props = (t.input_schema or {}).get("properties", {})
            req = set((t.input_schema or {}).get("required", []))
            args = ", ".join(f"{k}{'' if k in req else '?'}:{(v or {}).get('type', 'any')}"
                             for k, v in props.items())
            gate = "runs immediately" if t.policy == ALLOW else \
                "platform asks the user automatically — just call it"
            lines.append(f"- {t.name}({args}) [{gate}] — {t.description[:160]}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # execution
    # ------------------------------------------------------------------ #
    def validate(self, spec: ToolSpec, args: Any) -> dict:
        if not isinstance(args, dict):
            raise ToolError("arguments must be a JSON object")
        import jsonschema
        try:
            jsonschema.validate(args, spec.input_schema or {"type": "object"})
        except jsonschema.ValidationError as e:
            path = "/".join(str(p) for p in e.absolute_path) or "(root)"
            raise ToolError(f"invalid arguments for {spec.name} at {path}: {e.message}")
        except jsonschema.SchemaError:
            pass                                    # broken server schema: let the server judge
        return args

    @staticmethod
    def args_hash(spec_name: str, args: dict) -> str:
        return hashlib.sha256(f"{spec_name}|{json.dumps(args, sort_keys=True, default=str)}"
                              .encode()).hexdigest()

    async def _rate_limited(self, name: str) -> bool:
        from ..kv import MemoryKV, over_limit
        if not isinstance(self.kv, MemoryKV):          # shared across workers (Redis)
            return await over_limit(self.kv, f"tool:{name}", self.rate_per_minute)
        now, q = time.monotonic(), self._recent[name]   # single process: exact sliding window
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= self.rate_per_minute:
            return True
        q.append(now)
        return False

    async def call(self, spec: ToolSpec, args: dict, task_id: str = "") -> ToolResult:
        """Dispatch an ALREADY policy-approved call (the caller enforces allow/approve/deny).
        Never raises: every refusal or failure is a normalized ToolResult(ok=False) the
        model can read and react to (an exception here would kill the whole task)."""
        if spec.policy == DENY:
            return ToolResult(False, error=f"{spec.name} is denied by policy")
        key = f"{task_id}|{self.args_hash(spec.name, args)}"
        if not spec.read_only and key in self._idem:            # idempotency for writes
            prev = self._idem[key]
            return ToolResult(prev.ok, prev.text, prev.structured, prev.error, 0.0, cached=True)
        if task_id:
            self._task_calls[task_id] += 1
            if self._task_calls[task_id] > self.max_calls_per_task:
                return ToolResult(False, error=f"tool-call budget exhausted for this task "
                                               f"({self.max_calls_per_task}) — finish or stop")
        if await self._rate_limited(spec.name):
            return ToolResult(False, error=f"rate limit: {spec.name} called more than "
                                           f"{self.rate_per_minute}x in the last minute")
        t0 = time.monotonic()
        try:
            if spec.handler is not None:
                res = await asyncio.wait_for(spec.handler(args), self.timeout_s)
            else:
                conn = self._conns.get(spec.source.split(":", 1)[1])
                if conn is None:
                    raise ToolError(f"MCP server for {spec.name} is not configured")
                raw = await conn.call(spec.remote_name, args, self.timeout_s)
                res = self._normalize(raw)
        except ToolError as e:
            res = ToolResult(False, error=str(e))
        except asyncio.TimeoutError:
            res = ToolResult(False, error=f"timed out after {self.timeout_s:.0f}s")
        except Exception as e:                                   # noqa: BLE001
            res = ToolResult(False, error=f"{type(e).__name__}: {e}")
        res.elapsed_s = time.monotonic() - t0
        if not spec.read_only and res.ok:
            self._idem[key] = res
        return res

    @staticmethod
    def _normalize(raw) -> ToolResult:
        """MCP CallToolResult -> ToolResult (text for the model + structured content)."""
        parts = []
        for c in getattr(raw, "content", None) or []:
            ctype = getattr(c, "type", "")
            if ctype == "text":
                parts.append(c.text)
            elif ctype == "image":
                parts.append("[image]")
            elif ctype == "resource":
                res = getattr(c, "resource", None)
                parts.append(getattr(res, "text", "") or f"[resource {getattr(res, 'uri', '')}]")
            else:
                parts.append(f"[{ctype or 'content'}]")
        structured = getattr(raw, "structured_content", None)
        text = "\n".join(p for p in parts if p).strip()
        if not text and structured is not None:
            text = json.dumps(structured, default=str)
        if getattr(raw, "is_error", False):
            return ToolResult(False, text=text, structured=structured, error=text or "tool error")
        return ToolResult(True, text=text, structured=structured)

    def forget_task(self, task_id: str) -> None:
        self._task_calls.pop(task_id, None)
        for k in [k for k in self._idem if k.startswith(f"{task_id}|")]:
            self._idem.pop(k, None)
