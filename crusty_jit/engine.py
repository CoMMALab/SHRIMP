"""ctypes wrapper around crusty_compilations' libcrusty_jit_capi.so (the LLVM ORC JIT
+ Clang driver from crusty_compilations/src/jit).

crusty::jit::CompileOptions defaults to `-O3` (see crusty_compilations/include/jit/compiler.hh),
so every module compiled through this engine already gets full optimization; there is
no separate "-O3 mode" to opt into.
"""

import ctypes
import os
from typing import Optional


def _default_lib_path() -> str:
    env_override = os.environ.get("CRUSTY_JIT_LIB")
    if env_override:
        return env_override

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(repo_root, "crusty_compilations", "build", "libcrusty_jit_capi.so"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "Could not find libcrusty_jit_capi.so. Build it with:\n"
        "  cmake -S crusty_compilations -B crusty_compilations/build\n"
        "  cmake --build crusty_compilations/build --target crusty_jit_capi\n"
        "or point CRUSTY_JIT_LIB at an existing build of it."
    )


class CrustyJitEngine:
    """One JIT session (one LLJIT instance). Function pointers returned by lookup()
    stay valid only as long as this engine (and its underlying JitSession) is alive."""

    def __init__(self, lib_path: Optional[str] = None):
        self._lib = ctypes.CDLL(lib_path or _default_lib_path())

        self._lib.crusty_jit_create.restype = ctypes.c_void_p
        self._lib.crusty_jit_create.argtypes = []

        self._lib.crusty_jit_destroy.restype = None
        self._lib.crusty_jit_destroy.argtypes = [ctypes.c_void_p]

        self._lib.crusty_jit_compile.restype = ctypes.c_int
        self._lib.crusty_jit_compile.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]

        self._lib.crusty_jit_lookup.restype = ctypes.c_void_p
        self._lib.crusty_jit_lookup.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

        self._ctx = self._lib.crusty_jit_create()
        if not self._ctx:
            raise RuntimeError("crusty_jit_create() failed")

    def compile(self, source: str, module_id: str) -> bool:
        """Compiles `source` (a full C++ translation unit) at -O3 and adds it to this
        session. Returns True on a disk-cache hit, False on a fresh compile."""
        rc = self._lib.crusty_jit_compile(self._ctx, source.encode("utf-8"), module_id.encode("utf-8"))
        if rc < 0:
            raise RuntimeError(f"crusty_jit_compile failed with code {rc} for module '{module_id}'")
        return rc == 1

    def lookup(self, symbol: str) -> int:
        ptr = self._lib.crusty_jit_lookup(self._ctx, symbol.encode("utf-8"))
        if not ptr:
            raise RuntimeError(f"crusty_jit_lookup failed to find symbol '{symbol}'")
        return ptr

    def close(self) -> None:
        if getattr(self, "_ctx", None):
            self._lib.crusty_jit_destroy(self._ctx)
            self._ctx = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
