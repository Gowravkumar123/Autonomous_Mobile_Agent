"""
CLI:
    python -m onepiece_desktop pair --server https://host --code 123456
    python -m onepiece_desktop run [--window "Notepad"] [--allow notepad.exe] [--mouse]
    python -m onepiece_desktop stop | reset

Config + device token live in ~/.onepiece/desktop.json (override with ONEPIECE_HOME).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import socket
import sys
import threading

from .safety import DEFAULT_ALLOWED, HOME, Safety

CONFIG = HOME / "desktop.json"


def load() -> dict:
    try:
        return json.loads(CONFIG.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def save(cfg: dict) -> None:
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def default_device_id() -> str:
    host = "".join(ch if ch.isalnum() or ch == "-" else "-" for ch in socket.gethostname().lower())
    return f"desktop-{host}"[:80]


def pair(server: str, code: str, device_id: str) -> dict:
    import httpx
    r = httpx.post(server.rstrip("/") + "/v1/devices/pair",
                   json={"code": code, "device_id": device_id}, timeout=20)
    if r.status_code != 200:
        raise SystemExit(f"pairing failed ({r.status_code}): {r.text}")
    cfg = load()
    cfg.update({"server": server.rstrip("/"), "device_id": device_id, "token": r.json()["token"]})
    cfg.setdefault("allowed_processes", list(DEFAULT_ALLOWED))
    cfg.setdefault("workspaces", [str(HOME / "workspace")])
    save(cfg)
    return cfg


async def run(cfg: dict, args) -> None:
    from .agent import DesktopAgent
    from .uia import WindowsDesktop
    allowed = set(cfg.get("allowed_processes") or DEFAULT_ALLOWED) | set(args.allow or [])
    safety = Safety(allowed_processes=allowed,
                    workspaces=cfg.get("workspaces") or [str(HOME / "workspace")])
    for w in safety.workspaces:
        w.mkdir(parents=True, exist_ok=True)
    desktop = WindowsDesktop(safety, title_re=args.window or cfg.get("window_title_re", ""),
                             input_mode="mouse" if args.mouse else cfg.get("input_mode", "patterns"),
                             apps=cfg.get("apps") or {"notepad": "notepad.exe", "calculator": "calc.exe"})
    pending: dict = {}
    loop = asyncio.get_running_loop()

    def on_push(msg: dict) -> None:
        t = msg.get("type")
        if t == "approval_request":
            pending.update(kind="approval", task_id=msg.get("task_id"))
            print(f"\n[APPROVAL] {msg.get('title')}: {msg.get('summary')}\n  approve? [y/N] ", end="")
        elif t == "ask_user":
            pending.update(kind="answer", task_id=msg.get("task_id"))
            print(f"\n[QUESTION] {msg.get('questions') or msg.get('text')}  "
                  f"{msg.get('choices') or ''}\n  answer: ", end="")
        elif t == "handoff":
            pending.update(kind="handoff")
            print(f"\n[YOUR TURN] {msg.get('summary', '')}\n  {msg.get('message')}\n  "
                  "press Enter when done: ", end="")
        else:
            print(f"[{t}] {msg.get('message') or msg.get('detail') or msg.get('state') or ''}")
        sys.stdout.flush()

    agent = DesktopAgent(cfg["server"], cfg["device_id"], cfg["token"], desktop, safety,
                         on_push=on_push)

    def stdin_reader() -> None:
        for line in sys.stdin:
            kind = pending.pop("kind", None)
            tid = pending.pop("task_id", None)
            line = line.strip()
            if kind == "approval":
                asyncio.run_coroutine_threadsafe(
                    agent.approve(tid, line.lower() in ("y", "yes"), approver_note()), loop)
            elif kind == "answer":
                asyncio.run_coroutine_threadsafe(agent.answer(tid, line), loop)
            elif kind == "handoff":
                loop.call_soon_threadsafe(agent.resume)
    threading.Thread(target=stdin_reader, daemon=True).start()

    print(f"OnePiece desktop agent: {cfg['device_id']} -> {cfg['server']}\n"
          f"  allowlist: {', '.join(sorted(safety.allowed_processes))}\n"
          f"  workspace: {', '.join(map(str, safety.workspaces))}\n"
          f"  KILL SWITCH: move the mouse to the top-left corner (or `python -m "
          f"onepiece_desktop stop`)\n  action log: {safety.log_path}")
    try:
        await agent.run_forever()
    finally:
        desktop.close()
    if agent.auth_rejected:
        raise SystemExit("token rejected — run `pair` again")


def approver_note() -> str:
    return f"desktop console ({os.environ.get('USERNAME', 'user')})"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="onepiece_desktop")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pair", help="pair this PC with a 6-digit code from the admin")
    p.add_argument("--server", required=True)
    p.add_argument("--code", required=True)
    p.add_argument("--device-id", default=default_device_id())
    r = sub.add_parser("run", help="connect and accept tasks")
    r.add_argument("--window", help="regex of the window title to drive (default: foreground)")
    r.add_argument("--allow", action="append", help="extra allowlisted exe (repeatable)")
    r.add_argument("--mouse", action="store_true", help="allow real mouse/keyboard input")
    sub.add_parser("stop", help="engage the kill switch (creates the STOP file)")
    sub.add_parser("reset", help="clear the kill switch")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.cmd == "pair":
        cfg = pair(args.server, args.code, args.device_id)
        print(f"paired as {cfg['device_id']}; token saved to {CONFIG}")
    elif args.cmd == "run":
        cfg = load()
        if not cfg.get("token"):
            raise SystemExit("not paired — run `python -m onepiece_desktop pair ...` first")
        asyncio.run(run(cfg, args))
    elif args.cmd == "stop":
        Safety().stop_file.parent.mkdir(parents=True, exist_ok=True)
        Safety().stop_file.touch()
        print("kill switch engaged")
    elif args.cmd == "reset":
        Safety().reset_kill()
        print("kill switch cleared")


if __name__ == "__main__":
    main()
