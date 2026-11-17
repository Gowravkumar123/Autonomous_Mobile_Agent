"""
Local-dev compatibility shims.

On the deployment server these are no-ops: the real native libraries import fine.
They only activate when a native dependency can't load on a locked-down dev machine.

Concretely: LangGraph pulls in langchain-core -> langsmith, which imports `xxhash`
(a C extension) at module load. On the Windows dev box here, Application Control
blocks xxhash's `_xxhash.pyd`, so the import — and therefore LangGraph — fails. We
install a small pure-Python stand-in so the agent graph and its tests can run
locally. It is NEVER used when the real native module loads (i.e. on the server).
"""
from __future__ import annotations

import hashlib
import sys
import types


def _ensure_xxhash() -> None:
    try:
        import xxhash  # noqa: F401  real native module — always preferred
        return
    except Exception:
        pass  # blocked/absent -> install the pure-Python stand-in below

    module = types.ModuleType("xxhash")

    class _XX:
        """blake2b-backed stand-in for an xxhash hasher (local dev only)."""

        def __init__(self, data: bytes | str = b"", seed: int = 0, _bits: int = 64):
            self._bits = _bits
            self._seed = seed
            self.reset()
            if data:
                self.update(data)

        def reset(self) -> None:
            self._h = hashlib.blake2b(digest_size=max(8, self._bits // 8))
            if self._seed:
                self._h.update(repr(self._seed).encode())

        def update(self, data: bytes | str):
            if isinstance(data, str):
                data = data.encode()
            self._h.update(data)
            return self

        def digest(self) -> bytes:
            return self._h.digest()[: self._bits // 8]

        def hexdigest(self) -> str:
            return self.digest().hex()

        def intdigest(self) -> int:
            return int.from_bytes(self.digest(), "big")

        def copy(self):
            new = _XX(seed=self._seed, _bits=self._bits)
            new._h = self._h.copy()
            return new

        @property
        def digest_size(self) -> int:
            return self._bits // 8

    def _factory(bits: int):
        def make(data: bytes | str = b"", seed: int = 0):
            return _XX(data, seed, bits)
        return make

    for name, bits in (("xxh32", 32), ("xxh64", 64), ("xxh128", 128),
                       ("xxh3_64", 64), ("xxh3_128", 128)):
        fn = _factory(bits)
        setattr(module, name, fn)
        setattr(module, f"{name}_intdigest",
                (lambda f: lambda data=b"", seed=0: f(data, seed).intdigest())(fn))
        setattr(module, f"{name}_hexdigest",
                (lambda f: lambda data=b"", seed=0: f(data, seed).hexdigest())(fn))
        setattr(module, f"{name}_digest",
                (lambda f: lambda data=b"", seed=0: f(data, seed).digest())(fn))

    module.VERSION = "0.0.0-shim"
    module.XXHASH_VERSION = "0.0.0-shim"
    sys.modules["xxhash"] = module


_ensure_xxhash()
