"""FastAPI dependencies that enforce authentication on every protected route."""
from __future__ import annotations

from fastapi import Depends, HTTPException, Request, WebSocket

from .manager import AuthError, AuthManager, Principal

_OPEN = Principal("anonymous", "admin")   # used only when auth is disabled (no AUTH_SECRET)


def _bearer(value: str | None) -> str:
    if value and value.lower().startswith("bearer "):
        return value[7:].strip()
    return ""


def _auth(request_or_ws) -> AuthManager:
    return request_or_ws.app.state.auth


async def principal(request: Request) -> Principal:
    auth = _auth(request)
    if not auth.enabled:
        return _OPEN
    token = _bearer(request.headers.get("authorization"))
    if not token:
        raise HTTPException(401, "missing bearer token", headers={"WWW-Authenticate": "Bearer"})
    try:
        return auth.verify(token)
    except AuthError as e:
        raise HTTPException(401, str(e), headers={"WWW-Authenticate": "Bearer"})


async def admin(p: Principal = Depends(principal)) -> Principal:
    if not p.is_admin:
        raise HTTPException(403, "admin only")
    return p


def optional_principal(request: Request) -> Principal | None:
    """Best-effort identity for public endpoints that show more to signed-in callers."""
    auth = _auth(request)
    if not auth.enabled:
        return _OPEN
    token = _bearer(request.headers.get("authorization"))
    if not token:
        return None
    try:
        return auth.verify(token)
    except AuthError:
        return None


def ws_principal(ws: WebSocket) -> Principal | None:
    """Token from the Authorization header (preferred) or ?token= (browsers can't set WS
    headers). Returns None if missing/invalid — the caller must close the socket."""
    auth = _auth(ws)
    if not auth.enabled:
        return _OPEN
    token = _bearer(ws.headers.get("authorization")) or ws.query_params.get("token", "")
    if not token:
        return None
    try:
        return auth.verify(token)
    except AuthError:
        return None
