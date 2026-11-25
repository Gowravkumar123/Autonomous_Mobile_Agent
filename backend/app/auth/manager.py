"""
Authentication & device pairing (blueprint: API gateway & identity, device pairing).

Two kinds of signed tokens (HS256 JWT, key = AUTH_SECRET):
  - admin  : you / CLI / dashboard. Full access. Revoke by rotating AUTH_SECRET.
  - device : one paired phone, bound to its device_id. Can drive only its own socket and
             its own tasks. Revocable individually; re-pairing invalidates the old token.

Pairing: an admin asks for a short-lived 6-digit code; the phone submits it once and gets
its device token. Codes expire after 5 minutes and are single-use; 20 failed attempts in
10 minutes lock pairing entirely (brute-force guard: 1M codes vs ~20 guesses).

The device registry lives in Postgres when available (authoritative: a device token must
match the registry's current jti and not be revoked). Without a DB, device tokens are
checked by signature only, so a restart doesn't strand the phone.
"""
from __future__ import annotations

import hmac
import logging
import secrets
import time
import uuid
from dataclasses import dataclass, field

import jwt

log = logging.getLogger("agent.auth")

ALG = "HS256"
ISSUER = "onepiece"
PAIR_TTL_S = 300
PAIR_LOCK_FAILS = 20           # failed attempts (any code) ...
PAIR_LOCK_WINDOW_S = 600       # ... within this window locks pairing


class AuthError(Exception):
    pass


@dataclass(frozen=True)
class Principal:
    sub: str           # "admin" or a device_id
    scope: str         # "admin" | "device"
    jti: str = ""

    @property
    def is_admin(self) -> bool:
        return self.scope == "admin"

    def owns_device(self, device_id: str) -> bool:
        return self.is_admin or (self.scope == "device" and self.sub == device_id)


@dataclass
class _PairCode:
    code: str
    expires: float
    used: bool = False


@dataclass
class _Device:
    device_id: str
    jti: str
    revoked: bool = False
    paired_at: float = field(default_factory=time.time)


class AuthManager:
    def __init__(self, secret: str, db=None, token_days: int = 365):
        self.secret = (secret or "").strip()
        self.db = db
        self.token_days = token_days
        self._devices: dict[str, _Device] = {}
        self._codes: dict[str, _PairCode] = {}
        self._fails: list[float] = []

    @property
    def enabled(self) -> bool:
        return bool(self.secret)

    @property
    def registry_authoritative(self) -> bool:
        return bool(self.db and getattr(self.db, "ok", False))

    # ------------------------------------------------------------------ #
    # tokens
    # ------------------------------------------------------------------ #
    def _issue(self, sub: str, scope: str, days: int | None = None) -> tuple[str, str]:
        jti = uuid.uuid4().hex
        now = int(time.time())
        claims = {"iss": ISSUER, "sub": sub, "scope": scope, "iat": now, "jti": jti,
                  "exp": now + int((days or self.token_days) * 86400)}
        return jwt.encode(claims, self.secret, algorithm=ALG), jti

    def issue_admin(self, days: int | None = None) -> str:
        if not self.enabled:
            raise AuthError("AUTH_SECRET is not set")
        return self._issue("admin", "admin", days)[0]

    def verify(self, token: str) -> Principal:
        if not self.enabled:
            raise AuthError("auth disabled")
        try:
            c = jwt.decode(token, self.secret, algorithms=[ALG], issuer=ISSUER,
                           options={"require": ["exp", "sub", "scope", "jti"]})
        except jwt.ExpiredSignatureError:
            raise AuthError("token expired")
        except jwt.PyJWTError:
            raise AuthError("invalid token")
        scope = c.get("scope")
        if scope == "admin":
            return Principal("admin", "admin", c["jti"])
        if scope == "device":
            dev = self._devices.get(c["sub"])
            if dev is not None:
                # Known device: must be the CURRENT token (re-pairing replaces it) and live.
                if dev.revoked or not hmac.compare_digest(dev.jti, c["jti"]):
                    raise AuthError("device token revoked or superseded")
            elif self.registry_authoritative:
                raise AuthError("device not paired")
            # (no DB and unknown device, e.g. after a restart: signature is enough)
            return Principal(c["sub"], "device", c["jti"])
        raise AuthError("unknown scope")

    # ------------------------------------------------------------------ #
    # device registry
    # ------------------------------------------------------------------ #
    async def load(self) -> None:
        if not self.db:
            return
        try:
            for device_id, jti, revoked, paired_at in await self.db.load_devices():
                ts = paired_at.timestamp() if hasattr(paired_at, "timestamp") else time.time()
                self._devices[device_id] = _Device(device_id, jti, bool(revoked), ts)
            log.info("auth: %d paired device(s) loaded", len(self._devices))
        except Exception as e:
            log.warning("auth: could not load devices: %s", e)

    async def _register(self, device_id: str) -> str:
        token, jti = self._issue(device_id, "device")
        self._devices[device_id] = _Device(device_id, jti)
        if self.registry_authoritative:
            await self.db.upsert_device(device_id, jti)   # replaces (invalidates) any old token
        return token

    async def issue_device(self, device_id: str) -> str:
        """Admin-side direct issue (CLI / dashboard), same effect as a pairing."""
        if not self.enabled:
            raise AuthError("AUTH_SECRET is not set")
        return await self._register(device_id)

    async def revoke(self, device_id: str) -> bool:
        dev = self._devices.get(device_id)
        if dev is None:
            return False
        dev.revoked = True
        if self.registry_authoritative:
            await self.db.set_device_revoked(device_id, True)
        return True

    def list_devices(self) -> list[dict]:
        return [{"device_id": d.device_id, "revoked": d.revoked, "paired_at": d.paired_at}
                for d in sorted(self._devices.values(), key=lambda d: d.paired_at)]

    # ------------------------------------------------------------------ #
    # pairing
    # ------------------------------------------------------------------ #
    def start_pairing(self) -> tuple[str, int]:
        if not self.enabled:
            raise AuthError("AUTH_SECRET is not set")
        now = time.time()
        self._codes = {k: v for k, v in self._codes.items() if v.expires > now and not v.used}
        code = f"{secrets.randbelow(1_000_000):06d}"
        self._codes[code] = _PairCode(code, now + PAIR_TTL_S)
        return code, PAIR_TTL_S

    def _locked(self) -> bool:
        now = time.time()
        self._fails = [t for t in self._fails if now - t < PAIR_LOCK_WINDOW_S]
        return len(self._fails) >= PAIR_LOCK_FAILS

    async def complete_pairing(self, code: str, device_id: str) -> str:
        if not self.enabled:
            raise AuthError("auth disabled")
        if self._locked():
            raise AuthError("too many failed pairing attempts — try again later")
        device_id = (device_id or "").strip()
        if not device_id or len(device_id) > 80:
            raise AuthError("invalid device id")
        code = (code or "").strip()
        now = time.time()
        match = None
        for c in self._codes.values():                    # constant-time comparison
            if hmac.compare_digest(c.code, code):
                match = c
        if match is None or match.used or match.expires < now:
            self._fails.append(now)
            raise AuthError("invalid or expired pairing code")
        match.used = True
        token = await self._register(device_id)
        log.info("auth: paired device %s", device_id)
        return token
