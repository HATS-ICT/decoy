"""IPPO: independent PPO with parameter sharing inside each team.

Each team owns one actor-critic, shared by all of its agents. Teams learn
independently and treat each other as part of the environment, which is the
standard IPPO setup and a strong baseline on discrete-action PettingZoo tasks.

The implementation is deliberately small and self-contained (torch + numpy) so
it can serve as a reference for the environment rather than as a framework.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .networks import MaskedActorCritic
from .rollout import TeamBatch


@dataclass
class IPPOConfig:
    """Hyperparameters. Defaults are the common PPO settings, lightly tuned."""

    learning_rate: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 256
    hidden_size: int = 128
    normalize_advantages: bool = True
    # Stop an update early if the policy has already moved this far, measured
    # by approximate KL. Guards against a collapse on an unlucky batch.
    target_kl: Optional[float] = 0.03

    def to_dict(self) -> dict:
        return asdict(self)


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    segment_ends: List[int],
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generalised advantage estimation, computed within agent trajectories.

    ``segment_ends`` marks where one agent's trajectory stops and another's
    begins. Running the recursion across that boundary would bleed one agent's
    value estimate into another's, which is why the segments are tracked at all.
    """
    advantages = np.zeros_like(rewards, dtype=np.float64)
    start = 0
    for end in segment_ends:
        if end <= start:
            continue
        last_gae = 0.0
        for t in range(end - 1, start - 1, -1):
            non_terminal = 0.0 if dones[t] else 1.0
            next_value = values[t + 1] if (t + 1 < end and not dones[t]) else 0.0
            delta = rewards[t] + gamma * next_value * non_terminal - values[t]
            last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae
        start = end

    returns = advantages + values
    return advantages, returns


class IPPO:
    """Holds one :class:`MaskedActorCritic` per learning team, and updates them."""

    def __init__(
        self,
        obs_size: int,
        num_actions: int,
        teams=("T", "CT"),
        config: Optional[IPPOConfig] = None,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
    ):
        self.config = config or IPPOConfig()
        self.device = device or torch.device("cpu")
        self.teams = list(teams)
        if seed is not None:
            torch.manual_seed(seed)

        self.policies: Dict[str, MaskedActorCritic] = {
            team: MaskedActorCritic(obs_size, num_actions, self.config.hidden_size).to(self.device)
            for team in self.teams
        }
        self.optimizers = {
            team: torch.optim.Adam(policy.parameters(), lr=self.config.learning_rate, eps=1e-5)
            for team, policy in self.policies.items()
        }

    # ------------------------------------------------------------------ update
    def update(self, batches: Dict[str, TeamBatch]) -> Dict[str, dict]:
        """Run the PPO update for every learning team. Returns per-team metrics."""
        metrics = {}
        for team, batch in batches.items():
            if team not in self.policies or len(batch) == 0:
                continue
            metrics[team] = self._update_team(team, batch)
        return metrics

    def _update_team(self, team: str, batch: TeamBatch) -> dict:
        cfg = self.config
        policy = self.policies[team]
        optimizer = self.optimizers[team]

        rewards = np.asarray(batch.rewards, dtype=np.float64)
        values = np.asarray(batch.values, dtype=np.float64)
        dones = np.asarray(batch.dones, dtype=bool)
        advantages, returns = compute_gae(
            rewards, values, dones, batch.segment_ends, cfg.gamma, cfg.gae_lambda
        )

        obs = torch.as_tensor(np.asarray(batch.obs, dtype=np.float32), device=self.device)
        masks = torch.as_tensor(np.asarray(batch.masks), dtype=torch.bool, device=self.device)
        actions = torch.as_tensor(np.asarray(batch.actions), dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor(
            np.asarray(batch.log_probs, dtype=np.float32), device=self.device
        )
        advantages_t = torch.as_tensor(advantages.astype(np.float32), device=self.device)
        returns_t = torch.as_tensor(returns.astype(np.float32), device=self.device)

        n = len(batch)
        indices = np.arange(n)
        total_policy_loss = total_value_loss = total_entropy = 0.0
        total_kl = 0.0
        clip_fractions = []
        num_minibatches = 0
        stopped_early = False

        for _epoch in range(cfg.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, n, cfg.minibatch_size):
                mb = indices[start : start + cfg.minibatch_size]
                if len(mb) < 2:
                    continue
                mb_idx = torch.as_tensor(mb, dtype=torch.long, device=self.device)

                mb_adv = advantages_t[mb_idx]
                if cfg.normalize_advantages and len(mb) > 1:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                log_probs, entropy, new_values = policy.evaluate(
                    obs[mb_idx], masks[mb_idx], actions[mb_idx]
                )
                log_ratio = log_probs - old_log_probs[mb_idx]
                ratio = log_ratio.exp()

                with torch.no_grad():
                    # Schulman's low-variance approximate KL.
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    clip_fractions.append(
                        ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()
                    )

                policy_loss = -torch.min(
                    ratio * mb_adv,
                    torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef) * mb_adv,
                ).mean()
                value_loss = nn.functional.mse_loss(new_values, returns_t[mb_idx])
                entropy_loss = entropy.mean()

                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_loss.item()
                total_kl += approx_kl
                num_minibatches += 1

            if cfg.target_kl is not None and num_minibatches and \
                    (total_kl / num_minibatches) > cfg.target_kl:
                stopped_early = True
                break

        denom = max(num_minibatches, 1)
        explained = _explained_variance(values, returns)
        return {
            "policy_loss": total_policy_loss / denom,
            "value_loss": total_value_loss / denom,
            "entropy": total_entropy / denom,
            "approx_kl": total_kl / denom,
            "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else 0.0,
            "explained_variance": explained,
            "transitions": n,
            "stopped_early": stopped_early,
        }

    # ------------------------------------------------------------------ io
    def state_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "teams": self.teams,
            "policies": {t: p.state_dict() for t, p in self.policies.items()},
            "optimizers": {t: o.state_dict() for t, o in self.optimizers.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        for team, params in state["policies"].items():
            if team in self.policies:
                self.policies[team].load_state_dict(params)
        for team, params in state.get("optimizers", {}).items():
            if team in self.optimizers:
                self.optimizers[team].load_state_dict(params)


def _explained_variance(values: np.ndarray, returns: np.ndarray) -> float:
    """1 - Var(returns - values) / Var(returns); 0 means no better than the mean."""
    var_returns = np.var(returns)
    if var_returns == 0:
        return 0.0
    return float(1 - np.var(returns - values) / var_returns)
