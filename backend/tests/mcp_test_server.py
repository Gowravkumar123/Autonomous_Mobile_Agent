"""
A tiny MCP server (stdio) used by the tool-layer tests. Mix of read-only, write,
destructive and failing tools so every policy path can be exercised.

Run standalone:  python -m tests.mcp_test_server
"""
from __future__ import annotations

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

server = MCPServer("onepiece-test-tools")
_NOTES: dict[int, dict] = {}

RO = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True)


@server.tool(annotations=RO)
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@server.tool(annotations=RO)
def get_weather(city: str) -> dict:
    """Current weather for a city (canned test data)."""
    return {"city": city, "temp_c": 31, "condition": "Sunny", "humidity": 48}


@server.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False))
def create_note(title: str, body: str) -> str:
    """Create a note. Writes data (not read-only)."""
    nid = len(_NOTES) + 1
    _NOTES[nid] = {"title": title, "body": body}
    return f"note {nid} created: {title}"


@server.tool(annotations=RO)
def list_notes() -> list[dict]:
    """List all notes."""
    return [{"id": k, **v} for k, v in _NOTES.items()]


@server.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=True))
def delete_all_notes() -> str:
    """Delete every note. Destructive."""
    n = len(_NOTES)
    _NOTES.clear()
    return f"deleted {n} notes"


@server.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False))
def pay_invoice(invoice_id: str, amount: float) -> str:
    """Pay an invoice (fake). Name looks financial -> always needs approval."""
    return f"paid {amount} for {invoice_id}"


@server.tool(annotations=RO)
def broken() -> str:
    """Always fails (tests error handling)."""
    raise RuntimeError("upstream service unavailable")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == "--http":     # remote-style transport
        server.run("streamable-http", host="127.0.0.1", port=int(sys.argv[2]))
    else:
        server.run("stdio")
