"""Networks and observation normalisation for the IPPO baseline."""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class RunningNorm:
    """Running mean/variance normaliser (Welford), applied to observations.

    DECOY observations carry raw world coordinates (roughly -42..60) alongside
    healths (0..100) and a one-hot (0..1). Feeding that spread straight into an
    MLP makes the position dimensions dominate the first layer, so it is
    normalised online rather than with hand-set constants.
    """

    def __init__(self, size: int, epsilon: float = 1e-4):
        self.mean = np.zeros(size, dtype=np.float64)
        self.var = np.ones(size, dtype=np.float64)
        self.count = epsilon

    def update(self, batch: np.ndarray) -> None:
        batch = np.asarray(batch, dtype=np.float64)
        if batch.ndim == 1:
            batch = batch[None, :]
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]

        delta = batch_mean - self.mean
        total = self.count + batch_count

        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta**2 * self.count * batch_count / total) / total
        self.count = total

    def normalize(self, obs: np.ndarray, clip: float = 10.0) -> np.ndarray:
        normed = (np.asarray(obs, dtype=np.float64) - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(normed, -clip, clip).astype(np.float32)

    def state_dict(self) -> dict:
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load_state_dict(self, state: dict) -> None:
        self.mean = np.array(state["mean"], dtype=np.float64)
        self.var = np.array(state["var"], dtype=np.float64)
        self.count = float(state["count"])


def _init_layer(layer: nn.Linear, std: float = np.sqrt(2), bias: float = 0.0) -> nn.Linear:
    """Orthogonal init, the usual PPO default."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class MaskedActorCritic(nn.Module):
    """Shared-trunk actor-critic over a discrete action set with legality masks.

    The environment supplies an action mask per step (movement directions that
    have no neighbouring waypoint are illegal). Illegal logits are driven to
    -inf before the softmax, so they receive no probability and contribute no
    gradient, rather than being rejected after sampling.
    """

    def __init__(self, obs_size: int, num_actions: int, hidden_size: int = 128):
        super().__init__()
        self.num_actions = num_actions
        self.trunk = nn.Sequential(
            _init_layer(nn.Linear(obs_size, hidden_size)),
            nn.Tanh(),
            _init_layer(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
        )
        # Small final actor gain keeps the initial policy close to uniform.
        self.actor = _init_layer(nn.Linear(hidden_size, num_actions), std=0.01)
        self.critic = _init_layer(nn.Linear(hidden_size, 1), std=1.0)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor):
        hidden = self.trunk(obs)
        logits = self.actor(hidden)
        safe_mask = mask.bool()
        empty = ~safe_mask.any(dim=-1, keepdim=True)
        safe_mask = safe_mask | empty
        logits = logits.masked_fill(~safe_mask, torch.finfo(logits.dtype).min)
        return logits, self.critic(hidden).squeeze(-1)

    @torch.no_grad()
    def act(
        self, obs: torch.Tensor, mask: torch.Tensor, deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """Sample one action. Returns ``(action, log_prob, value)``."""
        logits, value = self.forward(obs, mask)
        dist = torch.distributions.Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        return int(action.item()), float(dist.log_prob(action).item()), float(value.item())

    def evaluate(
        self, obs: torch.Tensor, mask: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Log-probs, entropy and values for a batch, used by the PPO update."""
        logits, values = self.forward(obs, mask)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), values


class RandomPolicy:
    """Uniform over legal actions. The fixed opponent for the baseline runs."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, mask, deterministic: bool = False):
        legal = np.flatnonzero(np.asarray(mask))
        return int(self.rng.choice(legal)), 0.0, 0.0
