"""Import torch before Panda3D.

Panda3D initialises DLLs that make a later `import torch` fail on Windows with
OSError [WinError 1114]. pytest imports test modules in its own order, so the
guard in env/__init__.py is not always reached first.
"""
import torch  # noqa: F401
