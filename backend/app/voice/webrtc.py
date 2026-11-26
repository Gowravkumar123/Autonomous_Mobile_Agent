"""
WebRTC media transport for a voice session (blueprint: "for production audio, prefer
WebRTC media transport"). Optional: needs `aiortc` (+ PyAV). Signaling is one HTTP
offer/answer exchange (POST /v1/voice/{session_id}/webrtc); audio flows as Opus over
SRTP, and the same JSON control/events as the WebSocket ride a data channel named
"control". The session itself is unchanged — this only moves 20 ms PCM16 frames.
"""
from __future__ import annotations

import asyncio
import fractions
import json
import logging
import time

import numpy as np

from .audio import FRAME_BYTES, FRAME_SAMPLES, RATE

log = logging.getLogger("agent.voice.webrtc")


def available() -> bool:
    try:
        import aiortc  # noqa: F401
        import av      # noqa: F401  (blocked on some locked-down Windows boxes)
        return True
    except Exception:
        return False


def _make_track_class():
    import av
    from aiortc import MediaStreamTrack

    class SpeechTrack(MediaStreamTrack):
        """Outbound agent speech: paced 20 ms frames, silence when idle."""
        kind = "audio"

        def __init__(self):
            super().__init__()
            self.q: asyncio.Queue[bytes] = asyncio.Queue()
            self._pts = 0
            self._t0: float | None = None

        def clear(self) -> None:
            while not self.q.empty():
                self.q.get_nowait()

        async def recv(self):
            if self._t0 is None:
                self._t0 = time.monotonic()
            due = self._t0 + self._pts / RATE
            delay = due - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            try:
                pcm = self.q.get_nowait()
            except asyncio.QueueEmpty:
                pcm = b"\x00" * FRAME_BYTES
            frame = av.AudioFrame.from_ndarray(
                np.frombuffer(pcm, dtype="<i2").reshape(1, -1), format="s16", layout="mono")
            frame.sample_rate = RATE
            frame.pts = self._pts
            frame.time_base = fractions.Fraction(1, RATE)
            self._pts += FRAME_SAMPLES
            return frame

    return SpeechTrack


class WebRtcTransport:
    def __init__(self, session, ice_servers: list[str] | None = None):
        from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection
        self.session = session
        cfg = RTCConfiguration([RTCIceServer(urls=u) for u in (ice_servers or [])])
        self.pc = RTCPeerConnection(cfg)
        self.out = _make_track_class()()
        self.channel = None
        self._pending_json: list[str] = []
        self._readers: list[asyncio.Task] = []

        @self.pc.on("datachannel")
        def on_dc(ch):
            self.channel = ch
            for m in self._pending_json:
                ch.send(m)
            self._pending_json.clear()

            @ch.on("message")
            def on_msg(m):
                try:
                    asyncio.ensure_future(self.session.handle_json(json.loads(m)))
                except ValueError:
                    pass

        @self.pc.on("track")
        def on_track(track):
            if track.kind == "audio":
                self._readers.append(asyncio.ensure_future(self._read(track)))

        @self.pc.on("connectionstatechange")
        async def on_state():
            if self.pc.connectionState in ("failed", "closed"):
                await self.close()

    async def _read(self, track) -> None:
        import av
        resampler = av.AudioResampler(format="s16", layout="mono", rate=RATE)
        buf = bytearray()
        while True:
            try:
                frame = await track.recv()
            except Exception:
                return
            for f in resampler.resample(frame):
                buf += f.to_ndarray().astype("<i2").tobytes()
            while len(buf) >= FRAME_BYTES:
                await self.session.feed_audio(bytes(buf[:FRAME_BYTES]))
                del buf[:FRAME_BYTES]

    async def answer(self, offer_sdp: str, offer_type: str = "offer") -> dict:
        from aiortc import RTCSessionDescription
        await self.pc.setRemoteDescription(RTCSessionDescription(offer_sdp, offer_type))
        self.pc.addTrack(self.out)
        await self.pc.setLocalDescription(await self.pc.createAnswer())
        return {"sdp": self.pc.localDescription.sdp, "type": self.pc.localDescription.type}

    # ---- transport interface used by VoiceSession ----
    async def send_json(self, msg: dict) -> None:
        if msg.get("type") == "barge_in":
            self.out.clear()                 # drop queued speech immediately
        data = json.dumps(msg)
        if self.channel is not None and self.channel.readyState == "open":
            self.channel.send(data)
        else:
            self._pending_json.append(data)

    async def send_audio(self, frame: bytes) -> None:
        await self.out.q.put(frame)

    async def close(self) -> None:
        for r in self._readers:
            r.cancel()
        if self.session.transport is self:
            self.session.transport = None
        await self.pc.close()
