"""
Auth CLI — run on the server (reads AUTH_SECRET / DATABASE_URL from backend/.env).

  python -m app.auth.cli new-secret           # print a fresh random AUTH_SECRET
  python -m app.auth.cli admin-token [days]   # mint an admin bearer token
  python -m app.auth.cli device-token <id>    # issue a device token without pairing
  python -m app.auth.cli devices              # list paired devices
  python -m app.auth.cli revoke <id>          # revoke a device

Note: device-token/revoke write the registry in Postgres; a running server picks the
change up on its next restart (pair via the API for an immediate effect).
"""
from __future__ import annotations

import asyncio
import secrets
import sys

from ..config import get_settings
from .manager import AuthManager


async def _manager() -> AuthManager:
    s = get_settings()
    db = None
    if s.database_url.strip():
        from ..db.postgres import Database
        db = Database(s.database_url.strip())
        await db.init()
    mgr = AuthManager(s.auth_secret, db)
    await mgr.load()
    return mgr


async def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd = argv[0]
    if cmd == "new-secret":
        print(secrets.token_urlsafe(48))
        return 0
    mgr = await _manager()
    if not mgr.enabled:
        print("AUTH_SECRET is not set in backend/.env — run `new-secret` first.", file=sys.stderr)
        return 2
    try:
        if cmd == "admin-token":
            print(mgr.issue_admin(int(argv[1]) if len(argv) > 1 else None))
        elif cmd == "device-token" and len(argv) > 1:
            print(await mgr.issue_device(argv[1]))
        elif cmd == "devices":
            for d in mgr.list_devices():
                print(f"{d['device_id']:30} revoked={d['revoked']}")
        elif cmd == "revoke" and len(argv) > 1:
            print("revoked" if await mgr.revoke(argv[1]) else "unknown device")
        else:
            print(__doc__)
            return 2
        return 0
    finally:
        if mgr.db:
            await mgr.db.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv[1:])))
