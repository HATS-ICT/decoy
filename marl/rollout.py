"""Turning DECOY's AEC episodes into per-agent PPO trajectories.

The mapping is the subtle part of training on an AEC environment. Agents act
asynchronously, and ``env.last()`` reports the reward accrued *since that agent
last acted* -- so the reward you see when an agent is selected belongs to its
**previous** action, not the one you are about to take. Each agent therefore
carries one pending transition that is completed on its next turn (or when it
terminates), which is what :class:`_PendingTransition` tracks.

Getting this wrong silently shifts every reward by one agent-step, which is the
kind of bug that still trains, just worse.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch


@dataclass
class _PendingTransition:
    obs: np.ndarray
    action: int
    log_prob: float
    value: float
    mask: np.ndarray


@dataclass
class TeamBatch:
    """Flat arrays of completed transitions for one team."""

    obs: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    masks: List[np.ndarray] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    # Index boundaries between agent trajectories, so GAE never runs across
    # two different agents' experience.
    segment_ends: List[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.obs)

    def add(self, pending: _PendingTransition, reward: float, done: bool) -> None:
        self.obs.append(pending.obs)
        self.actions.append(pending.action)
        self.log_probs.append(pending.log_prob)
        self.values.append(pending.value)
        self.masks.append(pending.mask)
        self.rewards.append(reward)
        self.dones.append(done)

    def close_segment(self) -> None:
        if len(self.obs) and (not self.segment_ends or self.segment_ends[-1] != len(self.obs)):
            self.segment_ends.append(len(self.obs))


@dataclass
class RolloutStats:
    episodes: int = 0
    team_wins: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    win_reasons: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    returns: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    plants: int = 0
    kills: int = 0
    steps: int = 0

    def summary(self) -> dict:
        out = {
            "episodes": self.episodes,
            "rollout_steps": self.steps,
            "plant_rate": self.plants / max(self.episodes, 1),
            "kills_per_episode": self.kills / max(self.episodes, 1),
        }
        for team in ("T", "CT"):
            out[f"win_rate_{team}"] = self.team_wins.get(team, 0) / max(self.episodes, 1)
            values = self.returns.get(team, [])
            out[f"return_{team}"] = float(np.mean(values)) if values else 0.0
        out["win_reasons"] = dict(self.win_reasons)
        return out


def team_of(agent_id: str) -> str:
    return agent_id.split("_")[0]


def collect_rollout(
    env,
    policies: Dict[str, object],
    obs_norm,
    num_steps: int,
    learning_teams=("T", "CT"),
    device: Optional[torch.device] = None,
    max_episode_iters: int = 200_000,
):
    """Run the environment until ``num_steps`` transitions have been collected.

    Args:
        env: the wrapped PettingZoo AEC environment.
        policies: team name -> policy exposing ``act(obs, mask)``.
        obs_norm: :class:`~marl.networks.RunningNorm`, updated in place.
        num_steps: transitions to gather across all learning teams combined.
        learning_teams: only these teams' transitions are stored.

    Returns:
        ``(batches, stats)`` where ``batches`` maps team -> :class:`TeamBatch`.
    """
    device = device or torch.device("cpu")
    batches = {team: TeamBatch() for team in learning_teams}
    stats = RolloutStats()
    pending: Dict[str, Optional[_PendingTransition]] = {}
    episode_returns: Dict[str, float] = defaultdict(float)
    collected = 0

    # A fresh episode is started here rather than by the caller so that a
    # rollout boundary never lands mid-episode with transitions unaccounted for.
    env.reset()
    pending.clear()
    episode_returns.clear()
    raw_obs_buffer: List[np.ndarray] = []

    while collected < num_steps:
        for agent in env.agent_iter(max_iter=max_episode_iters):
            observation, reward, termination, truncation, info = env.last()
            team = team_of(agent)
            episode_returns[agent] += reward

            # Complete this agent's previous transition with the reward that
            # accrued while it was waiting.
            previous = pending.pop(agent, None)
            if previous is not None and team in batches:
                batches[team].add(previous, reward, termination or truncation)
                collected += 1

            if termination or truncation:
                if team in batches:
                    batches[team].close_segment()
                env.step(None)
                continue

            raw_obs_buffer.append(observation)
            normed = obs_norm.normalize(observation)
            mask = np.asarray(info["action_mask"])

            policy = policies[team]
            obs_tensor = torch.as_tensor(normed, dtype=torch.float32, device=device).unsqueeze(0)
            mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
            if hasattr(policy, "parameters"):
                action, log_prob, value = policy.act(obs_tensor, mask_tensor)
            else:
                action, log_prob, value = policy.act(normed, mask)

            env.step(action)
            stats.steps += 1
            if team in batches:
                pending[agent] = _PendingTransition(normed, action, log_prob, value, mask)

        # Episode finished: record outcome, then start the next one.
        winning_team, reason = env.unwrapped.round_outcome()
        stats.episodes += 1
        if winning_team is not None:
            stats.team_wins[winning_team.name] += 1
            stats.win_reasons[reason.name] += 1
        stats.plants += int(env.unwrapped.engine.bomb_status.name in ("Planted", "Defused", "Detonated"))
        stats.kills += sum(s["kills"] for s in env.unwrapped.agent_stats().values())

        per_team = defaultdict(list)
        for agent_id, value in episode_returns.items():
            per_team[team_of(agent_id)].append(value)
        for team, values in per_team.items():
            stats.returns[team].append(float(np.mean(values)))

        # Any transition still pending belongs to an agent that never got a
        # final turn; close its segment so GAE bootstraps rather than dropping it.
        for agent, previous in list(pending.items()):
            team = team_of(agent)
            if team in batches:
                batches[team].add(previous, 0.0, True)
                collected += 1
                batches[team].close_segment()
        pending.clear()

        for batch in batches.values():
            batch.close_segment()

        if collected < num_steps:
            env.reset()
            episode_returns.clear()

    if raw_obs_buffer:
        obs_norm.update(np.asarray(raw_obs_buffer, dtype=np.float64))

    for batch in batches.values():
        batch.close_segment()
    return batches, stats
