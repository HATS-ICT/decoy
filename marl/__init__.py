"""Multi-agent RL baselines for the DECOY environment.

Contains a self-contained IPPO (independent PPO with parameter sharing within a
team) implementation with action masking, which is the standard strong baseline
for discrete-action PettingZoo environments.

Deliberately dependency-free beyond torch/numpy: the point is a reference the
environment can be verified against, not a framework.
"""

import torch as _torch  # noqa: F401  (torch must precede Panda3D; see env/__init__.py)

__all__ = ["IPPO", "IPPOConfig", "MaskedActorCritic", "RunningNorm", "collect_rollout"]


def __getattr__(name):
    if name in ("IPPO", "IPPOConfig"):
        from . import ippo
        return getattr(ippo, name)
    if name in ("MaskedActorCritic", "RunningNorm"):
        from . import networks
        return getattr(networks, name)
    if name == "collect_rollout":
        from .rollout import collect_rollout
        return collect_rollout
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
