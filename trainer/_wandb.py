"""Lazy handle for Weights & Biases.

Every wandb call in this package is already guarded by ``--use_wandb``, but the
modules imported it at the top level, which made an optional logging backend a
hard requirement for training the damage models at all.

This proxy keeps the ``wandb.log(...)`` call sites unchanged while deferring the
import until something actually touches an attribute.
"""


class _LazyWandb:
    def __getattr__(self, name):
        try:
            import wandb as _wandb
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "Weights & Biases logging was requested but wandb is not installed. "
                "Install it with `pip install wandb`, or drop --use_wandb."
            ) from exc
        return getattr(_wandb, name)


wandb = _LazyWandb()
