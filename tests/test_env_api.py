"""PettingZoo API conformance and core-semantics tests for the DECOY environment.

These are deliberately end-to-end: each one constructs the real engine (which
loads the 75MB map and both damage checkpoints), so the suite is slow but it
exercises the paths that actually broke.

Panda3D allows only one ``ShowBase`` per process and ``CSGOEngine`` subclasses
it, so **environments cannot coexist** -- every test creates its env through
:func:`make_env` and closes it before the next one starts.

Run with ``pytest tests/ -v``.
"""

from contextlib import contextmanager

import numpy as np
import pytest
from pettingzoo.test import api_test

from env.csgo_environment import env, observation_size, NUM_ACTIONS
from env.rewards import RewardConfig, SPARSE_REWARD_CONFIG


@contextmanager
def make_env(**kwargs):
    """Create an environment and guarantee it is released afterwards."""
    kwargs.setdefault("num_team_agents", 2)
    kwargs.setdefault("render_mode", None)
    e = env(**kwargs)
    try:
        yield e
    finally:
        e.close()


def random_rollout(e, seed=0, max_iter=20000):
    """Play one episode with a masked-random policy; return per-agent returns."""
    e.reset(seed=seed)
    rng = np.random.default_rng(seed)
    returns = {a: 0.0 for a in e.possible_agents}
    steps = 0
    for agent in e.agent_iter(max_iter=max_iter):
        _obs, reward, termination, truncation, info = e.last()
        returns[agent] += reward
        if termination or truncation:
            action = None
        else:
            action = int(rng.choice(np.where(info["action_mask"])[0]))
        e.step(action)
        steps += 1
    return returns, steps


# --------------------------------------------------------------------- spaces
@pytest.mark.parametrize("n,expected", [(1, 12), (2, 16), (5, 28)])
def test_observation_size_formula(n, expected):
    assert observation_size(n) == expected


def test_observation_size_rejects_empty_roster():
    with pytest.raises(ValueError):
        observation_size(0)


def test_observation_matches_declared_space():
    """Regression: the env declared shape (3,) while emitting 16 values."""
    with make_env(seed=0) as e:
        e.reset(seed=0)
        for agent in e.possible_agents:
            space = e.observation_space(agent)
            obs = e.observe(agent)
            assert obs.shape == space.shape, f"{agent}: {obs.shape} vs {space.shape}"
            assert space.contains(obs), f"{agent}: observation outside declared bounds"


def test_action_mask_is_int8_and_stop_always_legal():
    """gymnasium's Discrete.sample(mask=...) rejects anything but int8."""
    with make_env(seed=0) as e:
        e.reset(seed=0)
        _obs, _r, _t, _tr, info = e.last()
        mask = info["action_mask"]
        assert mask.dtype == np.int8
        assert mask.shape == (NUM_ACTIONS,)
        assert mask[8] == 1, "the stop action must always be legal"


# ------------------------------------------------------------------- api test
def test_pettingzoo_api_conformance():
    with make_env(seed=0) as e:
        api_test(e, num_cycles=3, verbose_progress=False)


# -------------------------------------------------------------------- rewards
def test_rewards_are_not_identically_zero():
    """Regression: Agent.current_reward was never written, so returns were 0."""
    with make_env(seed=0) as e:
        returns, _steps = random_rollout(e, seed=0)
    assert any(abs(v) > 1e-9 for v in returns.values()), "every return was exactly zero"


def test_winning_team_outranks_losing_team():
    """The terminal reward must dominate: winners end above losers."""
    with make_env(seed=0) as e:
        returns, _steps = random_rollout(e, seed=0)
        winning_team, _reason = e.unwrapped.round_outcome()
    assert winning_team is not None, "round did not resolve"

    winners = [v for a, v in returns.items() if a.split("_")[0] == winning_team.name]
    losers = [v for a, v in returns.items() if a.split("_")[0] != winning_team.name]
    assert min(winners) > max(losers), f"winners {winners} did not beat losers {losers}"


def test_sparse_config_yields_only_terminal_reward():
    with make_env(num_team_agents=1, reward_config=SPARSE_REWARD_CONFIG) as e:
        returns, _steps = random_rollout(e, seed=3)
    for agent, value in returns.items():
        assert value == pytest.approx(1.0) or value == pytest.approx(-1.0), \
            f"{agent} got {value}, expected a pure win/loss reward"


def test_reward_config_validates_team_spirit():
    assert RewardConfig(team_spirit=1.0).team_spirit == 1.0
    with pytest.raises(ValueError):
        RewardConfig(team_spirit=1.5)


def test_damage_produces_kill_and_death_credit():
    """Two agents spawned within sight of each other must actually fight."""
    with make_env(num_team_agents=1, seed=1) as e:
        raw = e.unwrapped
        wp = raw.engine.waypoints
        e.reset(seed=1)

        anchor = raw.engine.agents["T_0"].current_waypoint["id"]
        anchor_pos = wp.get_position(anchor)
        near = wp.get_nearest_waypoint(anchor_pos + type(anchor_pos)(1.5, 0, 0), return_id=True)

        e.reset(seed=1, options={"player_spawns": {
            "T_0": {"init_waypoint_id": anchor},
            "CT_0": {"init_waypoint_id": near},
        }})
        rng = np.random.default_rng(1)
        for agent in e.agent_iter(max_iter=4000):
            _obs, _r, termination, truncation, info = e.last()
            action = None if (termination or truncation) else \
                int(rng.choice(np.where(info["action_mask"])[0]))
            e.step(action)

        stats = raw.agent_stats()
        total_damage = sum(s["damage_dealt"] for s in stats.values())
        total_kills = sum(s["kills"] for s in stats.values())

    assert total_damage > 0, "no damage was dealt between agents in line of sight"
    assert total_kills > 0, "damage was dealt but no kill was credited"


# ------------------------------------------------------------------- lifecycle
def test_all_agents_retire_by_end_of_episode():
    """Regression: dead agents were never removed, so agent_iter never ended."""
    with make_env(seed=1) as e:
        random_rollout(e, seed=1)
        assert e.agents == [], f"agents left over: {e.agents}"


def test_stepping_a_terminated_agent_with_an_action_raises():
    with make_env(seed=0) as e:
        e.reset(seed=0)
        for _agent in e.agent_iter(max_iter=20000):
            _obs, _r, termination, truncation, info = e.last()
            if termination or truncation:
                with pytest.raises(ValueError):
                    e.step(0)
                return
            e.step(int(np.where(info["action_mask"])[0][0]))
    pytest.fail("no agent ever terminated")


def test_env_can_be_recreated_after_close():
    """Panda3D allows one ShowBase at a time; close() must release it."""
    with make_env(num_team_agents=1) as e:
        e.reset(seed=0)
    with make_env(num_team_agents=1) as e:
        e.reset(seed=0)


# ---------------------------------------------------------------------- seeding
def test_same_seed_reproduces_episode():
    """Regression: reset(seed=...) accepted a seed and ignored it entirely."""
    def trace(seed):
        with make_env() as e:
            e.reset(seed=seed)
            rng = np.random.default_rng(123)
            out = []
            for agent in e.agent_iter(max_iter=300):
                obs, reward, termination, truncation, info = e.last()
                out.append((agent, round(float(obs.sum()), 4), round(reward, 6)))
                action = None if (termination or truncation) else \
                    int(rng.choice(np.where(info["action_mask"])[0]))
                e.step(action)
            return out

    assert trace(7) == trace(7), "same seed produced different episodes"
    assert trace(7) != trace(8), "different seeds produced identical episodes"


# ----------------------------------------------------------------- spawn modes
def test_fixed_spawn_by_world_position():
    """Regression: this path raised UnboundLocalError on reset_waypoint."""
    with make_env(num_team_agents=1, seed=2) as e:
        e.reset(seed=2)
        anchor = e.unwrapped.engine.agents["T_0"].current_waypoint["id"]
        pos = e.unwrapped.engine.waypoints.get_position(anchor)

        e.reset(seed=2, options={"player_spawns": {
            "T_0": {"init_pos": (pos.x, pos.y, pos.z)},
        }})
        spawned = e.unwrapped.engine.agents["T_0"].position
        assert abs(spawned.x - pos.x) < 1.0 and abs(spawned.y - pos.y) < 1.0


def test_fixed_spawn_by_waypoint_id():
    with make_env(num_team_agents=1, seed=2) as e:
        e.reset(seed=2)
        target = list(e.unwrapped.engine.waypoints.graph.nodes)[100]
        e.reset(seed=2, options={"player_spawns": {"CT_0": {"init_waypoint_id": target}}})
        assert e.unwrapped.engine.agents["CT_0"].current_waypoint["id"] == target


# ------------------------------------------------------------------- roster size
@pytest.mark.parametrize("n", [1, 3])
def test_non_default_roster_sizes_run(n):
    with make_env(num_team_agents=n, seed=0) as e:
        returns, steps = random_rollout(e, seed=0)
        assert steps > 0
        assert len(returns) == 2 * n
        assert e.unwrapped.round_outcome()[0] is not None
