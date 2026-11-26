"""
Local safety controls. These run ON the desktop, independent of the backend: even a
compromised or buggy brain cannot make this machine act outside them.

- Process allowlist: only windows owned by allowlisted executables are observed or driven.
  Anything else is reported as a blocked screen with no content and no screenshot.
- Workspace boundary: text that looks like a file path must resolve inside a workspace
  folder (stops "save to C:\\Windows\\..." or "open ~/.ssh/id_rsa").
- Kill switch: move the mouse into the top-left screen corner, or create the STOP file —
  every later action is refused and the backend cancels the task. Latches until reset.
- Action log: every action (and refusal) is appended to a local JSONL file. Typed text in
  password fields is never written.
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

HOME = Path(os.environ.get("ONEPIECE_HOME", Path.home() / ".onepiece"))

DEFAULT_ALLOWED = ("notepad.exe", "calc.exe", "calculatorapp.exe", "mspaint.exe",
                   "wordpad.exe", "write.exe")

# Drive paths (C:\..., C:/...), UNC (\\server\share), home (~/..), absolute POSIX (/etc/..)
# and parent traversal. Deliberately broad: a false positive just asks for a workspace path.
_PATH_RE = re.compile(r"(?:^|[\s\"'])(?:[a-zA-Z]:[\\/]|\\\\|~[\\/]|/(?:[^/\s]+/)+|\.\.[\\/])")


def cursor_pos() -> tuple[int, int]:
    pt = ctypes.wintypes.POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


@dataclass
class Safety:
    allowed_processes: set[str] = field(default_factory=lambda: set(DEFAULT_ALLOWED))
    workspaces: list[Path] = field(default_factory=lambda: [HOME / "workspace"])
    log_path: Path = field(default_factory=lambda: HOME / "desktop-actions.jsonl")
    stop_file: Path = field(default_factory=lambda: HOME / "STOP")
    corner_px: int = 2
    cursor: Callable[[], tuple[int, int]] = cursor_pos      # injectable for tests
    _killed: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self.allowed_processes = {p.lower() for p in self.allowed_processes}
        self.workspaces = [Path(w).expanduser().resolve() for w in self.workspaces]

    # ---- allowlist ---------------------------------------------------------
    def process_allowed(self, exe: str) -> bool:
        return (exe or "").lower() in self.allowed_processes

    # ---- workspace boundary ------------------------------------------------
    def looks_like_path(self, text: str) -> bool:
        return bool(_PATH_RE.search(text or ""))

    def path_allowed(self, text: str) -> bool:
        """True if `text` is not a path, or every path-like token resolves in a workspace."""
        if not self.looks_like_path(text):
            return True
        tokens = [t.strip("\"'") for t in re.split(r"\s+(?=[a-zA-Z]:[\\/]|\\\\|~|/|\.\.)", text)]
        for tok in tokens:
            if not self.looks_like_path(" " + tok):
                continue
            try:
                p = Path(os.path.expanduser(tok)).resolve()
            except (OSError, ValueError):
                return False
            if not any(p == w or w in p.parents for w in self.workspaces):
                return False
        return True

    # ---- kill switch -------------------------------------------------------
    def check_kill(self) -> str:
        """Return the reason the kill switch is engaged ('' if not). Latches."""
        if self._killed:
            return self._killed
        reason = ""
        if self.stop_file.exists():
            reason = f"STOP file present ({self.stop_file})"
        else:
            try:
                x, y = self.cursor()
                if x <= self.corner_px and y <= self.corner_px:
                    reason = "mouse moved to the top-left corner"
            except Exception:
                pass
        if reason:
            self._killed = reason
            self.log({"event": "kill_switch", "reason": reason})
        return self._killed

    def reset_kill(self) -> None:
        self._killed = ""
        try:
            self.stop_file.unlink()
        except FileNotFoundError:
            pass

    # ---- action log --------------------------------------------------------
    def log(self, entry: dict) -> None:
        entry = {"ts": round(time.time(), 3), **entry}
        with self._lock:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
