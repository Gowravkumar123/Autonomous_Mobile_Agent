"""
Audio plumbing for the voice gateway. One internal format everywhere:
PCM16 little-endian, mono, 16 kHz, in 20 ms frames (640 bytes).

VAD (blueprint 4.1: "detect speech start/end; tune end-of-turn silence thresholds") is a
small adaptive-energy detector — no native code, so it runs anywhere (the dev box blocks
webrtcvad's DLL). Clients should capture with platform echo cancellation (Android
VOICE_COMMUNICATION + AcousticEchoCanceler, browser getUserMedia echoCancellation), which
is what makes barge-in over the agent's own speech reliable.
"""
from __future__ import annotations

import io
import math
import wave
from collections import deque
from dataclasses import dataclass, field

import numpy as np

RATE = 16000
FRAME_MS = 20
FRAME_SAMPLES = RATE * FRAME_MS // 1000          # 320
FRAME_BYTES = FRAME_SAMPLES * 2                   # 640


def frames(pcm: bytes) -> list[bytes]:
    """Split PCM16 into whole 20 ms frames (a trailing partial frame is zero-padded)."""
    out = [pcm[i:i + FRAME_BYTES] for i in range(0, len(pcm), FRAME_BYTES)]
    if out and len(out[-1]) < FRAME_BYTES:
        out[-1] = out[-1] + b"\x00" * (FRAME_BYTES - len(out[-1]))
    return out


def rms_dbfs(frame: bytes) -> float:
    x = np.frombuffer(frame, dtype="<i2").astype(np.float32)
    if x.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(x * x)))
    return 20 * math.log10(max(rms, 1e-3) / 32768.0)


def pcm_to_wav(pcm: bytes, rate: int = RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def wav_to_pcm16k(data: bytes) -> bytes:
    """Decode a PCM WAV (any rate, mono/stereo, 16-bit) to PCM16 mono 16 kHz."""
    with wave.open(io.BytesIO(data), "rb") as w:
        ch, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"unsupported sample width {width}")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return resample(x, rate, RATE).astype("<i2").tobytes()


def resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst or x.size == 0:
        return x
    n = int(round(x.size * dst / src))
    t = np.linspace(0, x.size - 1, n)
    return np.interp(t, np.arange(x.size), x)


def tone(ms: int, freq: float = 220.0, db: float = -12.0, rate: int = RATE) -> bytes:
    """Test/feedback helper: a sine tone as PCM16."""
    n = rate * ms // 1000
    amp = 32767 * 10 ** (db / 20)
    t = np.arange(n) / rate
    return (amp * np.sin(2 * math.pi * freq * t)).astype("<i2").tobytes()


@dataclass
class VadEvent:
    kind: str                 # "start" | "end" | "discard"
    pcm: bytes = b""          # the utterance (with pre-roll) on "end"
    speech_ms: int = 0


@dataclass
class Vad:
    """Adaptive energy VAD with hangover.

    A frame is speech if it is `margin_db` above the tracked noise floor AND above an
    absolute floor. Speech STARTS after `start_frames` consecutive speech frames and ENDS
    after `end_silence_ms` of non-speech (the end-of-turn threshold). Utterances shorter
    than `min_speech_ms` (coughs, clicks) are discarded. `pre_roll_ms` of audio before
    the start is kept so the first syllable isn't clipped.
    """
    margin_db: float = 10.0
    abs_floor_db: float = -50.0
    start_frames: int = 3
    end_silence_ms: int = 700
    min_speech_ms: int = 250
    max_utterance_ms: int = 30000
    pre_roll_ms: int = 300
    noise_db: float = -60.0
    speaking: bool = False
    bias_db: float = 0.0                 # raised while the agent talks (echo guard)
    _run: int = 0
    _silence: int = 0
    _speech_frames: int = 0
    _buf: bytearray = field(default_factory=bytearray)
    _pre: deque = field(default_factory=deque)
    floor_window_ms: int = 1500
    _hist: deque = field(default_factory=deque)

    def is_speech(self, db: float) -> bool:
        return db > max(self.noise_db + self.margin_db + self.bias_db, self.abs_floor_db)

    def feed(self, frame: bytes) -> VadEvent | None:
        db = rms_dbfs(frame)
        # Minimum-statistics noise floor: the quietest frame of the last ~1.5 s. Speech
        # always has dips between syllables so it never raises the floor, but a steady
        # new background (a fan switching on) does — so it can't hold a turn open forever.
        self._hist.append(db)
        while len(self._hist) > self.floor_window_ms // FRAME_MS:
            self._hist.popleft()
        self.noise_db = max(min(self._hist), -90.0)
        voiced = self.is_speech(db)
        if not self.speaking:
            self._pre.append(frame)
            while len(self._pre) > max(1, self.pre_roll_ms // FRAME_MS):
                self._pre.popleft()
            self._run = self._run + 1 if voiced else 0
            if self._run >= self.start_frames:
                self.speaking = True
                self._buf = bytearray(b"".join(self._pre))
                self._pre.clear()
                self._speech_frames = self._run
                self._silence = 0
                return VadEvent("start")
            return None
        self._buf += frame
        if voiced:
            self._speech_frames += 1
            self._silence = 0
        else:
            self._silence += FRAME_MS
        too_long = len(self._buf) // FRAME_BYTES * FRAME_MS >= self.max_utterance_ms
        if self._silence >= self.end_silence_ms or too_long:
            return self._finish()
        return None

    def force_end(self) -> VadEvent | None:
        """Push-to-talk release / stream end."""
        return self._finish() if self.speaking else None

    def _finish(self) -> VadEvent:
        speech_ms = self._speech_frames * FRAME_MS
        pcm = bytes(self._buf)
        self.speaking, self._run, self._silence, self._speech_frames = False, 0, 0, 0
        self._buf = bytearray()
        if speech_ms < self.min_speech_ms:
            return VadEvent("discard", speech_ms=speech_ms)
        return VadEvent("end", pcm=pcm, speech_ms=speech_ms)
