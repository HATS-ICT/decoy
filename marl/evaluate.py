"""Evaluate a trained checkpoint against a fixed random opponent.

    python -m marl.evaluate runs/ippo-T-vs-random --episodes 30

Reports the random-policy baseline alongside the trained policy, in both
stochastic (sampled) and greedy (argmax) modes, because the two can behave
quite differently on a navigation task: a deterministic policy that reaches a
dead end has no way to break out of it.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from env.csgo_environment import env as make_env, observation_size, NUM_ACTIONS
from env.rewards import RewardConfig
from marl.ippo import IPPO, IPPOConfig
from marl.networks import RunningNorm, RandomPolicy


def run_episodes(environment, policies, obs_norm, episodes, seed, deterministic):
    """Play ``episodes`` rounds and return aggregate statistics."""
    wins = defaultdict(int)
    reasons = defaultdict(int)
    plants = 0
    returns = defaultdict(list)
    decisions = []
    kills = defaultdict(list)

    for ep in range(episodes):
        environment.reset(seed=seed + ep)
        episode_returns = defaultdict(float)
        steps = 0
        for agent in environment.agent_iter(max_iter=500_000):
            observation, reward, termination, truncation, info = environment.last()
            episode_returns[agent] += reward
            if termination or truncation:
                environment.step(None)
                continue

            team = agent.split("_")[0]
            policy = policies[team]
            mask = np.asarray(info["action_mask"])
            if hasattr(policy, "parameters"):
                normed = obs_norm.normalize(observation)
                obs_t = torch.as_tensor(normed, dtype=torch.float32).unsqueeze(0)
                mask_t = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
                action, _, _ = policy.act(obs_t, mask_t, deterministic=deterministic)
            else:
                action, _, _ = policy.act(observation, mask)
            environment.step(action)
            steps += 1

        team, reason = environment.unwrapped.round_outcome()
        if team is not None:
            wins[team.name] += 1
            reasons[reason.name] += 1
        plants += int(environment.unwrapped.engine.bomb_status.name
                      in ("Planted", "Defused", "Detonated"))
        decisions.append(steps)
        stats = environment.unwrapped.agent_stats()
        for agent_id, value in episode_returns.items():
            returns[agent_id.split("_")[0]].append(value)
        for agent_id, s in stats.items():
            kills[agent_id.split("_")[0]].append(s["kills"])

    return {
        "episodes": episodes,
        "win_rate_T": wins["T"] / episodes,
        "win_rate_CT": wins["CT"] / episodes,
        "plant_rate": plants / episodes,
        "return_T": float(np.mean(returns["T"])),
        "return_CT": float(np.mean(returns["CT"])),
        "decisions_per_episode": float(np.mean(decisions)),
        "kills_T": float(np.mean(kills["T"])),
        "win_reasons": dict(reasons),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=50_000)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    checkpoint = torch.load(args.run_dir / "checkpoint.pt", map_location="cpu",
                            weights_only=False)
    config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    train_args = checkpoint["args"]
    team_size = train_args["team_size"]
    learn_team = train_args["learn_team"]

    obs_size = observation_size(team_size)
    algo = IPPO(obs_size, NUM_ACTIONS,
                teams=config["learning_teams"],
                config=IPPOConfig(**config["ippo"]))
    algo.load_state_dict(checkpoint["algo"])
    obs_norm = RunningNorm(obs_size)
    obs_norm.load_state_dict(checkpoint["obs_norm"])

    environment = make_env(
        num_team_agents=team_size,
        render_mode=None,
        reward_config=RewardConfig(**config["reward"]),
        seed=args.seed,
    )

    trained = {t: p for t, p in algo.policies.items()}
    for team in ("T", "CT"):
        trained.setdefault(team, RandomPolicy(seed=args.seed + 1))
    all_random = {team: RandomPolicy(seed=args.seed + 1) for team in ("T", "CT")}

    results = {}
    print(f"evaluating {args.run_dir} over {args.episodes} episodes "
          f"({team_size}v{team_size}, learning team {learn_team})\n")

    for label, policies, deterministic in [
        ("random baseline", all_random, False),
        ("trained (stochastic)", trained, False),
        ("trained (greedy)", trained, True),
    ]:
        stats = run_episodes(environment, policies, obs_norm,
                             args.episodes, args.seed, deterministic)
        results[label] = stats
        print(f"{label}")
        print(f"  win rate {learn_team:<3}      : {stats[f'win_rate_{learn_team}']:.2f}")
        print(f"  plant rate        : {stats['plant_rate']:.2f}")
        print(f"  mean return T     : {stats['return_T']:+.3f}")
        print(f"  decisions/episode : {stats['decisions_per_episode']:.0f}")
        print(f"  kills/agent T     : {stats['kills_T']:.2f}")
        print(f"  outcomes          : {stats['win_reasons']}")
        print()

    environment.close()

    out = args.out or (args.run_dir / "evaluation.json")
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return results


if __name__ == "__main__":
    main()
