"""
MCP client connections (official SDK, protocol v2 client).

Each configured server gets one MCPConnection. The SDK's transports and ClientSession are
anyio context managers that must be entered AND exited by the same task, so every
connection runs in its own long-lived "owner" task that opens the transport + session,
lists tools, then parks until stop() is requested. Tool calls from any request task go
through the shared session. reconnect() = stop the owner, start a fresh one.

Transports:
  - stdio:            {"command": "...", "args": [...], "env": {...}, "cwd": "..."}
  - streamable HTTP:  {"url": "https://host/mcp", "headers": {"Authorization": "..."}}
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import AsyncExitStack
from typing import Any

log = logging.getLogger("agent.mcp")


class MCPConnection:
    def __init__(self, name: str, cfg: dict, timeout_s: float = 30.0):
        self.name = name
        self.cfg = cfg
        self.timeout_s = timeout_s
        self.session = None
        self.tools: list = []
        self.error: str = ""
        self.server_name: str = ""
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None

    @property
    def connected(self) -> bool:
        return self.session is not None

    # ------------------------------------------------------------------ #
    async def start(self) -> bool:
        self._ready, self._stop = asyncio.Event(), asyncio.Event()
        self.error = ""
        self._task = asyncio.create_task(self._run(), name=f"mcp:{self.name}")
        try:
            await asyncio.wait_for(self._ready.wait(), self.timeout_s)
        except asyncio.TimeoutError:
            self.error = f"timed out connecting after {self.timeout_s:.0f}s"
            await self.stop()
        if self.error:
            log.warning("MCP %s unavailable: %s", self.name, self.error)
        else:
            log.info("MCP %s connected (%s): %d tools", self.name, self.server_name, len(self.tools))
        return self.connected

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, 10)
            except (asyncio.TimeoutError, Exception):
                self._task.cancel()
        self._task = None
        self.session = None

    async def reconnect(self) -> bool:
        await self.stop()
        return await self.start()

    async def _run(self) -> None:
        from mcp import ClientSession
        try:
            async with AsyncExitStack() as stack:
                read, write = await self._open_transport(stack)
                session = await stack.enter_async_context(
                    ClientSession(read, write, read_timeout_seconds=self.timeout_s))
                init = await session.initialize()
                self.server_name = getattr(getattr(init, "server_info", None), "name", "") or ""
                self.tools = await self._list_all_tools(session)
                self.session = session
                self._ready.set()
                await self._stop.wait()
        except Exception as e:                       # noqa: BLE001 — report, never crash the app
            self.error = f"{type(e).__name__}: {e}"
        finally:
            self.session = None
            self._ready.set()

    async def _open_transport(self, stack: AsyncExitStack):
        cfg = self.cfg
        if cfg.get("command"):
            from mcp import StdioServerParameters
            from mcp.client.stdio import get_default_environment, stdio_client
            env = get_default_environment()
            env.update({k: str(v) for k, v in (cfg.get("env") or {}).items()})
            command = cfg["command"]
            if command in ("python", "python3"):
                command = sys.executable          # the backend's own interpreter/venv
            params = StdioServerParameters(command=command, args=list(cfg.get("args") or []),
                                           env=env, cwd=cfg.get("cwd") or os.getcwd())
            return await stack.enter_async_context(stdio_client(params))
        if cfg.get("url"):
            from mcp.client.streamable_http import (create_mcp_http_client,
                                                    streamable_http_client)
            http = await stack.enter_async_context(
                create_mcp_http_client(headers=dict(cfg.get("headers") or {})))
            return await stack.enter_async_context(streamable_http_client(cfg["url"],
                                                                          http_client=http))
        raise ValueError("server config needs either 'command' (stdio) or 'url' (HTTP)")

    @staticmethod
    async def _list_all_tools(session) -> list:
        from mcp import types
        tools, cursor = [], None
        for _ in range(20):                          # paginate (bounded)
            params = types.PaginatedRequestParams(cursor=cursor) if cursor else None
            res = await session.list_tools(params=params)
            tools.extend(res.tools)
            cursor = getattr(res, "next_cursor", None)
            if not cursor:
                break
        return tools

    # ------------------------------------------------------------------ #
    async def call(self, tool: str, args: dict[str, Any], timeout_s: float):
        if self.session is None and not await self.reconnect():
            raise ConnectionError(f"MCP server '{self.name}' is not connected: {self.error}")
        try:
            return await self.session.call_tool(tool, args, read_timeout_seconds=timeout_s)
        except (ConnectionError, OSError, EOFError) as e:
            log.warning("MCP %s call failed (%s) — reconnecting once", self.name, e)
            if not await self.reconnect():
                raise
            return await self.session.call_tool(tool, args, read_timeout_seconds=timeout_s)
