"""Unit tests for the IPPO baseline.

These are fast: they exercise the algorithm pieces (GAE, masking, normalisation)
without constructing the simulation. The end-to-end training smoke test lives in
``test_marl_integration.py``.
"""

import numpy as np
import pytest
import torch

from marl.ippo import IPPOConfig, compute_gae, _explained_variance
from marl.networks import MaskedActorCritic, RandomPolicy, RunningNorm
from marl.rollout import TeamBatch, _PendingTransition, team_of


# ------------------------------------------------------------------------ GAE
def test_gae_on_single_terminal_segment():
    """With gamma=1, lambda=1 the advantage is just return-to-go minus value."""
    rewards = np.array([1.0, 1.0, 1.0])
    values = np.zeros(3)
    dones = np.array([False, False, True])
    adv, ret = compute_gae(rewards, values, dones, [3], gamma=1.0, gae_lambda=1.0)
    np.testing.assert_allclose(adv, [3.0, 2.0, 1.0])
    np.testing.assert_allclose(ret, [3.0, 2.0, 1.0])


def test_gae_does_not_leak_across_agent_segments():
    """Two agents' trajectories must not bleed into each other.

    Both segments are identical, so if the recursion respected the boundary the
    advantages must be identical too.
    """
    rewards = np.array([1.0, 1.0, 1.0, 1.0])
    values = np.zeros(4)
    dones = np.array([False, True, False, True])
    adv, _ = compute_gae(rewards, values, dones, [2, 4], gamma=1.0, gae_lambda=1.0)
    np.testing.assert_allclose(adv[:2], adv[2:])
    np.testing.assert_allclose(adv, [2.0, 1.0, 2.0, 1.0])


def test_gae_discounts():
    rewards = np.array([0.0, 0.0, 1.0])
    values = np.zeros(3)
    dones = np.array([False, False, True])
    adv, _ = compute_gae(rewards, values, dones, [3], gamma=0.5, gae_lambda=1.0)
    np.testing.assert_allclose(adv, [0.25, 0.5, 1.0])


def test_gae_bootstraps_value_at_a_non_terminal_cut():
    """A segment that ends without done=True should use the next value estimate."""
    rewards = np.array([0.0, 0.0])
    values = np.array([0.0, 4.0])
    dones = np.array([False, False])
    adv, _ = compute_gae(rewards, values, dones, [2], gamma=1.0, gae_lambda=1.0)
    # t=1 has nothing after it inside the segment, so it bootstraps to 0: -4.
    # t=0 sees value[1]=4, so delta = 0 + 4 - 0 = 4, plus the propagated -4.
    np.testing.assert_allclose(adv, [0.0, -4.0])


def test_explained_variance():
    values = np.array([1.0, 2.0, 3.0])
    assert _explained_variance(values, values) == pytest.approx(1.0)
    assert _explained_variance(np.zeros(3), values) == pytest.approx(0.0)
    # Constant targets carry no variance to explain.
    assert _explained_variance(np.zeros(3), np.ones(3)) == 0.0


# -------------------------------------------------------------------- masking
def test_masked_policy_never_samples_an_illegal_action():
    torch.manual_seed(0)
    net = MaskedActorCritic(obs_size=16, num_actions=9, hidden_size=32)
    obs = torch.randn(1, 16)
    mask = torch.zeros(1, 9, dtype=torch.bool)
    mask[0, [2, 7]] = True

    for _ in range(50):
        action, log_prob, _value = net.act(obs, mask)
        assert action in (2, 7), f"sampled illegal action {action}"
        assert np.isfinite(log_prob)


def test_masked_logits_put_zero_probability_on_illegal_actions():
    torch.manual_seed(0)
    net = MaskedActorCritic(obs_size=16, num_actions=9, hidden_size=32)
    obs = torch.randn(4, 16)
    mask = torch.zeros(4, 9, dtype=torch.bool)
    mask[:, 0] = True
    mask[:, 5] = True

    logits, _values = net(obs, mask)
    probs = torch.softmax(logits, dim=-1)
    illegal = probs[:, [1, 2, 3, 4, 6, 7, 8]]
    assert torch.all(illegal < 1e-20), f"illegal actions kept probability {illegal.max()}"
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(4))


def test_all_illegal_mask_does_not_produce_nan():
    """Guard path: a degenerate mask must not poison the softmax."""
    net = MaskedActorCritic(obs_size=16, num_actions=9, hidden_size=32)
    obs = torch.randn(2, 16)
    mask = torch.zeros(2, 9, dtype=torch.bool)
    logits, values = net(obs, mask)
    probs = torch.softmax(logits, dim=-1)
    assert torch.isfinite(probs).all()
    assert torch.isfinite(values).all()


def test_evaluate_shapes():
    net = MaskedActorCritic(obs_size=16, num_actions=9, hidden_size=32)
    obs = torch.randn(7, 16)
    mask = torch.ones(7, 9, dtype=torch.bool)
    actions = torch.randint(0, 9, (7,))
    log_probs, entropy, values = net.evaluate(obs, mask, actions)
    assert log_probs.shape == (7,)
    assert entropy.shape == (7,)
    assert values.shape == (7,)


def test_random_policy_respects_the_mask():
    policy = RandomPolicy(seed=0)
    mask = np.zeros(9, dtype=np.int8)
    mask[[3, 8]] = 1
    for _ in range(50):
        action, _lp, _v = policy.act(np.zeros(16), mask)
        assert action in (3, 8)


# -------------------------------------------------------------- normalisation
def test_running_norm_matches_batch_statistics():
    rng = np.random.default_rng(0)
    data = rng.normal(loc=5.0, scale=3.0, size=(1000, 4))
    norm = RunningNorm(4)
    for chunk in np.array_split(data, 10):
        norm.update(chunk)
    np.testing.assert_allclose(norm.mean, data.mean(axis=0), rtol=1e-6)
    np.testing.assert_allclose(norm.var, data.var(axis=0), rtol=1e-3)


def test_running_norm_clips_and_round_trips():
    norm = RunningNorm(3)
    norm.update(np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]))
    out = norm.normalize(np.array([1e9, 1.0, -1e9]), clip=5.0)
    assert out.max() <= 5.0 and out.min() >= -5.0
    assert out.dtype == np.float32

    restored = RunningNorm(3)
    restored.load_state_dict(norm.state_dict())
    np.testing.assert_allclose(restored.mean, norm.mean)
    np.testing.assert_allclose(restored.var, norm.var)


# ------------------------------------------------------------------- batching
def test_team_batch_segments():
    batch = TeamBatch()
    pending = _PendingTransition(np.zeros(4), 0, 0.0, 0.0, np.ones(9, dtype=np.int8))
    batch.close_segment()          # no-op while empty
    assert batch.segment_ends == []

    batch.add(pending, 1.0, False)
    batch.add(pending, 1.0, True)
    batch.close_segment()
    batch.close_segment()          # idempotent
    assert batch.segment_ends == [2]
    assert len(batch) == 2


def test_team_of():
    assert team_of("T_0") == "T"
    assert team_of("CT_3") == "CT"


def test_ippo_config_roundtrip():
    cfg = IPPOConfig(learning_rate=1e-3, gamma=0.9)
    as_dict = cfg.to_dict()
    assert as_dict["learning_rate"] == 1e-3
    assert as_dict["gamma"] == 0.9
    assert IPPOConfig(**as_dict) == cfg
