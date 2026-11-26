"""Voice session registry: ownership, reconnects, idle cleanup, turn metadata."""
from __future__ import annotations

import time
from collections import deque
from typing import Callable

from .session import Backend, VoiceConfig, VoiceSession

IDLE_TTL_S = 1800


class VoiceManager:
    def __init__(self, make_backend: Callable[[str, str], Backend], db=None):
        self.make_backend = make_backend          # (owner, device_id) -> Backend
        self.db = db
        self.sessions: dict[str, VoiceSession] = {}
        self._seen: dict[str, float] = {}
        self.recent_turns: deque = deque(maxlen=500)

    def record_turn(self, turn: dict) -> None:
        self.recent_turns.append(turn)
        if self.db is not None:
            self.db.enqueue_voice_turn(turn)

    async def get_or_create(self, session_id: str, owner: str, device_id: str,
                            cfg: VoiceConfig | None = None) -> VoiceSession | None:
        """None if the id belongs to someone else (no hijacking a conversation)."""
        await self._gc()
        s = self.sessions.get(session_id)
        if s is not None:
            if s.owner != owner:
                return None
            if cfg is not None:
                s.cfg.mode = cfg.mode
        else:
            s = VoiceSession(session_id, owner, self.make_backend(owner, device_id), cfg)
            await s.start()
            self.sessions[session_id] = s
        self._seen[session_id] = time.monotonic()
        return s

    def touch(self, session_id: str) -> None:
        self._seen[session_id] = time.monotonic()

    async def _gc(self) -> None:
        now = time.monotonic()
        for sid, seen in list(self._seen.items()):
            s = self.sessions.get(sid)
            if s is not None and s.transport is None and now - seen > IDLE_TTL_S:
                await s.close()
                self.sessions.pop(sid, None)
                self._seen.pop(sid, None)

    async def close_all(self) -> None:
        for s in self.sessions.values():
            await s.close()
        self.sessions.clear()
