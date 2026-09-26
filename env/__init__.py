"""DECOY: a data-driven discretized CS:GO simulation environment.

Importing this package pulls in ``torch`` before any Panda3D module. That order
is load-bearing on Windows: Panda3D initialises DLLs that make a subsequent
``import torch`` fail with ``OSError: [WinError 1114]``. Importing torch first
is harmless everywhere else, so it is done unconditionally.
"""

import torch as _torch  # noqa: F401  (import order matters -- see module docstring)

__all__ = ["env", "raw_env", "CSGOEngine", "WaypointGraph"]


def __getattr__(name):
    # Lazily re-export the heavy symbols so `import env` stays cheap.
    if name in ("env", "raw_env"):
        from . import csgo_environment
        return getattr(csgo_environment, name)
    if name == "CSGOEngine":
        from .game_engine import CSGOEngine
        return CSGOEngine
    if name == "WaypointGraph":
        from .waypoints import WaypointGraph
        return WaypointGraph
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
