"""PettingZoo AEC environment wrapping the DECOY CS:GO simulation.

Agents act asynchronously: the simulation runs until an agent reaches its target
waypoint and requests a decision, so ``agent_selection`` follows the engine's
decision queue rather than a fixed turn order. That is a legitimate AEC
ordering, but it means the reward and termination bookkeeping has to be driven
from the engine on every step, which is what :meth:`raw_env._sync_from_engine`
does.
"""

import functools
from typing import Dict, List, Optional

from gymnasium.spaces import Discrete, Box
import numpy as np

from pettingzoo import AECEnv
from pettingzoo.utils import wrappers

from .game_engine import CSGOEngine
from .rewards import RewardConfig
from .config import AGENT_MAX_HEALTH
from .utils import BombStatus

#: Movement directions plus the explicit "stop" action.
NUM_ACTIONS = 9


def observation_size(num_team_agents: int) -> int:
    """Length of the flat observation vector for a given roster size.

    Layout, in order::

        own position            3
        own health              1
        bomb position           3
        bomb status one-hot     len(BombStatus)
        teammate positions      3 * (num_team_agents - 1)
        teammate healths        1 * (num_team_agents - 1)

    The environment used to advertise a fixed ``shape=(3,)`` regardless of this,
    so anything sizing a network or a replay buffer from the declared space got
    it wrong for every roster size.
    """
    if num_team_agents < 1:
        raise ValueError(f"num_team_agents must be >= 1, got {num_team_agents}")
    teammates = num_team_agents - 1
    return 3 + 1 + 3 + len(BombStatus) + 3 * teammates + teammates


def env(num_team_agents=5, render_mode=None, debug_mode=False, show_waypoints=False,
        show_minimap=False, reward_config: Optional[RewardConfig] = None,
        seed: Optional[int] = None):
    """Construct the environment behind the standard PettingZoo wrapper stack."""
    environment = raw_env(
        num_team_agents, render_mode=render_mode, debug_mode=debug_mode,
        show_waypoints=show_waypoints, show_minimap=show_minimap,
        reward_config=reward_config, seed=seed,
    )
    environment = wrappers.AssertOutOfBoundsWrapper(environment)
    environment = wrappers.OrderEnforcingWrapper(environment)
    return environment


class raw_env(AECEnv):
    metadata = {
        "render_modes": ["spectator"],
        "name": "decoy_csgo_v1",
        "is_parallelizable": False,
    }

    def __init__(self, num_team_agents, render_mode=None, debug_mode=False,
                 show_waypoints=False, show_minimap=False,
                 reward_config: Optional[RewardConfig] = None,
                 seed: Optional[int] = None):
        super().__init__()
        self.num_team_agents = num_team_agents
        self.possible_agents = [f"{team}_{i}" for team in ["T", "CT"] for i in range(num_team_agents)]
        self.agents = self.possible_agents[:]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        self.render_mode = render_mode
        self._observation_size = observation_size(num_team_agents)
        self._action_space = Discrete(NUM_ACTIONS)

        self.engine = CSGOEngine(
            num_team_agents, render_mode=render_mode, debug_mode=debug_mode,
            show_waypoints=show_waypoints, show_minimap=show_minimap,
            reward_config=reward_config, seed=seed,
        )
        self._observation_space = self._build_observation_space()

        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.agent_selection = None
        self.state_sequence = self._empty_state_sequence()

    # ------------------------------------------------------------------ spaces
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_space

    def _build_observation_space(self) -> Box:
        """Per-dimension bounds, derived from the loaded map rather than +/-inf.

        Position components are bounded by the waypoint graph's bounding box
        (with a margin for the spawn height offset and for agents drifting off a
        waypoint between decisions), healths by [0, AGENT_MAX_HEALTH] and the
        bomb-status one-hot by [0, 1].
        """
        lo_xyz, hi_xyz = self.engine.waypoints.position_bounds()
        margin = 5.0
        lo_xyz = lo_xyz - margin
        hi_xyz = hi_xyz + margin
        max_hp = float(AGENT_MAX_HEALTH)

        low, high = [], []

        def add(lows, highs):
            low.extend(lows)
            high.extend(highs)

        add(lo_xyz, hi_xyz)                              # own position
        add([0.0], [max_hp])                             # own health
        add(lo_xyz, hi_xyz)                              # bomb position
        add([0.0] * len(BombStatus), [1.0] * len(BombStatus))
        for _ in range(self.num_team_agents - 1):        # teammate positions
            add(lo_xyz, hi_xyz)
        for _ in range(self.num_team_agents - 1):        # teammate healths
            add([0.0], [max_hp])

        assert len(low) == self._observation_size, (
            f"observation bounds ({len(low)}) disagree with observation_size "
            f"({self._observation_size})"
        )
        return Box(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------ helpers
    def _empty_state_sequence(self):
        return {
            agent: {
                "observation": [],
                "reward": [],
                "termination": [],
                "truncation": [],
                "episode_length": 0,
            }
            for agent in self.possible_agents
        }

    def _sync_from_engine(self) -> None:
        """Pull rewards, termination flags and action masks out of the engine.

        Rewards are *drained*: each call returns what accrued since the previous
        one, which is exactly the AEC notion of "reward since this agent last
        acted" once ``_accumulate_rewards`` folds it into ``_cumulative_rewards``.
        """
        drained = self.engine.collect_rewards()
        for agent in self.agents:
            engine_agent = self.engine.agents[agent]
            self.rewards[agent] = float(drained.get(agent, 0.0))
            self.terminations[agent] = bool(engine_agent.termination)
            self.truncations[agent] = False
            self.infos[agent] = {"action_mask": engine_agent.action_mask}

    def _select_next_agent(self) -> Optional[str]:
        """Next agent to act, taken from the engine queue.

        Skips agents that have already been retired, since a dead agent can
        still be sitting in the decision queue from before it died.
        """
        while True:
            candidate = self.engine.get_next_agent()
            if candidate is None:
                break
            if candidate in self.agents:
                return candidate

        # Queue drained. Any agent still flagged terminated has not been
        # retired yet and must be handed back so the caller can step it once
        # with None.
        pending = [a for a in self.agents if self.terminations[a] or self.truncations[a]]
        if pending:
            return pending[0]
        return self.agents[0] if self.agents else None

    def _retire_agent(self, agent: str) -> None:
        """Remove a terminated agent, mirroring ``AECEnv._was_dead_step``.

        Upstream's helper also picks the next ``agent_selection`` from its own
        ordering, which would fight the engine's decision queue, so the removal
        bookkeeping is done here and selection stays with the engine.
        """
        del self.terminations[agent]
        del self.truncations[agent]
        del self.rewards[agent]
        del self._cumulative_rewards[agent]
        del self.infos[agent]
        self.agents.remove(agent)

    # ------------------------------------------------------------------ API
    def observe(self, agent):
        """Return the observation for ``agent``.

        Pure: reward and termination state are maintained by ``step`` and
        ``reset`` rather than being written as a side effect of observing.
        """
        return self.engine.agents[agent].observation

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.engine.seed(seed)

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.engine.reset(options)
        self._sync_from_engine()
        self.agent_selection = self._select_next_agent()
        self.state_sequence = self._empty_state_sequence()

    def step(self, action):
        """Advance the simulation by one agent decision.

        A terminated agent is stepped once with ``None`` and then retired, which
        is what lets ``agent_iter`` end: it stops when ``self.agents`` is empty.
        """
        agent = self.agent_selection
        if agent is None:
            return

        if self.terminations[agent] or self.truncations[agent]:
            if action is not None:
                raise ValueError(
                    f"{agent} has terminated; the only valid action is None, got {action!r}"
                )
            self._retire_agent(agent)
            self._clear_rewards()
            self.agent_selection = self._select_next_agent()
            return

        # The agent's accumulated reward was handed to the caller by last(), so
        # it starts over from here.
        self._cumulative_rewards[agent] = 0.0

        self.engine.set_move_target(agent, action)
        self.engine.update_simulation()

        self._sync_from_engine()
        self._accumulate_rewards()
        self.agent_selection = self._select_next_agent()

    def render(self):
        if self.render_mode is None:
            return
        # Rendering is driven by the engine's own frame pacing inside
        # update_simulation(); there is no separate frame to emit here.
        return None

    def close(self):
        self.engine.destroy()

    # ------------------------------------------------------------------ extras
    @property
    def reward_config(self) -> RewardConfig:
        return self.engine.reward_config

    def agent_stats(self) -> Dict[str, dict]:
        """Per-agent episode statistics (kills, damage, distance travelled)."""
        return {aid: a.stats.to_dict() for aid, a in self.engine.agents.items()}

    def round_outcome(self):
        """``(winning_team, win_reason)`` for the finished round, or ``(None, None)``."""
        return self.engine.winning_team, self.engine.winning_reason
