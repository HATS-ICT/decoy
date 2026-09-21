"""Train an IPPO baseline on the DECOY environment.

Examples::

    # Train T against a fixed random CT. Win rate is the clearest evidence of
    # learning, since a random T never wins at all.
    python -m marl.train --opponent random --total-steps 200000

    # Both teams learn simultaneously.
    python -m marl.train --opponent selfplay --total-steps 400000

Results land in ``runs/<name>/``: ``metrics.csv``, ``config.json`` and
``checkpoint.pt``.

Note: the environment cannot be vectorised in-process -- Panda3D permits one
ShowBase per process and CSGOEngine subclasses it -- so rollouts are collected
from a single environment instance.
"""

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from env.csgo_environment import env as make_env, observation_size, NUM_ACTIONS
from env.rewards import RewardConfig, DEFAULT_REWARD_CONFIG, SHAPED_REWARD_CONFIG, SPARSE_REWARD_CONFIG
from marl.ippo import IPPO, IPPOConfig
from marl.networks import RunningNorm, RandomPolicy
from marl.rollout import collect_rollout

REWARD_PRESETS = {
    "shaped": SHAPED_REWARD_CONFIG,
    "dense": DEFAULT_REWARD_CONFIG,
    "sparse": SPARSE_REWARD_CONFIG,
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--total-steps", type=int, default=200_000,
                   help="total environment transitions to train on")
    p.add_argument("--rollout-steps", type=int, default=4096,
                   help="transitions collected between PPO updates")
    p.add_argument("--team-size", type=int, default=2, help="agents per team")
    p.add_argument("--opponent", choices=["random", "selfplay"], default="random",
                   help="'random' trains T against a fixed random CT; "
                        "'selfplay' trains both teams")
    p.add_argument("--learn-team", choices=["T", "CT"], default="T",
                   help="which team learns when --opponent random")
    p.add_argument("--reward", choices=list(REWARD_PRESETS), default="shaped",
                   help="reward preset (see env/rewards.py)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--name", type=str, default=None, help="run directory name")
    p.add_argument("--out-dir", type=str, default="runs")
    p.add_argument("--eval-episodes", type=int, default=10,
                   help="greedy evaluation episodes at the end of training")
    return p.parse_args(argv)


def build_run_dir(args) -> Path:
    name = args.name or f"ippo-{args.opponent}-{args.team_size}v{args.team_size}-s{args.seed}"
    run_dir = Path(args.out_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def evaluate(environment, policies, obs_norm, episodes: int, seed: int) -> dict:
    """Evaluate both stochastic and greedy action selection.

    Both are reported because they can differ sharply here: a deterministic
    policy that walks into a dead end has no way out, and ends up oscillating
    between adjacent waypoints until the round times out.
    """
    from marl.evaluate import run_episodes
    return {
        "stochastic": run_episodes(environment, policies, obs_norm, episodes,
                                   seed + 10_000, deterministic=False),
        "greedy": run_episodes(environment, policies, obs_norm, episodes,
                               seed + 10_000, deterministic=True),
    }


def main(argv=None):
    args = parse_args(argv)
    run_dir = build_run_dir(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    reward_config: RewardConfig = REWARD_PRESETS[args.reward]
    learning_teams = ("T", "CT") if args.opponent == "selfplay" else (args.learn_team,)

    environment = make_env(
        num_team_agents=args.team_size,
        render_mode=None,
        reward_config=reward_config,
        seed=args.seed,
    )

    obs_size = observation_size(args.team_size)
    config = IPPOConfig(
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        entropy_coef=args.entropy_coef,
        hidden_size=args.hidden_size,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
    )
    algo = IPPO(obs_size, NUM_ACTIONS, teams=learning_teams, config=config, seed=args.seed)
    obs_norm = RunningNorm(obs_size)

    policies = dict(algo.policies)
    for team in ("T", "CT"):
        if team not in policies:
            policies[team] = RandomPolicy(seed=args.seed + 1)

    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump({
            "args": vars(args),
            "ippo": config.to_dict(),
            "reward": asdict(reward_config),
            "obs_size": obs_size,
            "num_actions": NUM_ACTIONS,
            "learning_teams": list(learning_teams),
        }, f, indent=2)

    metrics_path = run_dir / "metrics.csv"
    metrics_file = metrics_path.open("w", newline="", encoding="utf-8")
    writer = None

    print(f"run dir       : {run_dir}")
    print(f"observation   : {obs_size} dims, {NUM_ACTIONS} actions")
    print(f"learning teams: {list(learning_teams)}  (opponent: {args.opponent})")
    print(f"reward preset : {args.reward}")
    print()
    header = (f"{'steps':>8} {'iter':>5} {'winT':>6} {'winCT':>6} {'plant':>6} "
              f"{'retT':>8} {'retCT':>8} {'entropy':>8} {'evar':>7} {'sps':>6}")
    print(header)
    print("-" * len(header))

    total_steps = 0
    iteration = 0
    start_time = time.time()

    try:
        while total_steps < args.total_steps:
            iteration += 1
            t0 = time.time()
            batches, stats = collect_rollout(
                environment, policies, obs_norm, args.rollout_steps,
                learning_teams=learning_teams,
            )
            collect_time = time.time() - t0
            total_steps += stats.steps

            update_metrics = algo.update(batches)
            summary = stats.summary()

            primary = learning_teams[0]
            entropy = update_metrics.get(primary, {}).get("entropy", float("nan"))
            evar = update_metrics.get(primary, {}).get("explained_variance", float("nan"))
            sps = stats.steps / max(collect_time, 1e-9)

            row = {
                "iteration": iteration,
                "wall_time": round(time.time() - start_time, 1),
                "steps_per_second": round(sps, 1),
                **{k: v for k, v in summary.items() if k != "win_reasons"},
                # Set last: summary carries its own per-rollout step count, and
                # spreading it after this would silently clobber the total.
                "steps": total_steps,
                "win_reasons": json.dumps(summary["win_reasons"]),
            }
            for team, m in update_metrics.items():
                for key, value in m.items():
                    row[f"{team}_{key}"] = value

            if writer is None:
                writer = csv.DictWriter(metrics_file, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            metrics_file.flush()

            print(f"{total_steps:>8} {iteration:>5} "
                  f"{summary['win_rate_T']:>6.2f} {summary['win_rate_CT']:>6.2f} "
                  f"{summary['plant_rate']:>6.2f} "
                  f"{summary['return_T']:>8.3f} {summary['return_CT']:>8.3f} "
                  f"{entropy:>8.3f} {evar:>7.2f} {sps:>6.0f}")
    finally:
        metrics_file.close()

    checkpoint = {
        "algo": algo.state_dict(),
        "obs_norm": obs_norm.state_dict(),
        "args": vars(args),
    }
    torch.save(checkpoint, run_dir / "checkpoint.pt")

    print("\nevaluating (greedy)...")
    eval_stats = evaluate(environment, policies, obs_norm, args.eval_episodes, args.seed)
    with (run_dir / "eval.json").open("w", encoding="utf-8") as f:
        json.dump(eval_stats, f, indent=2)
    print(json.dumps(eval_stats, indent=2))

    environment.close()
    print(f"\ndone in {time.time()-start_time:.0f}s -> {run_dir}")
    return eval_stats


if __name__ == "__main__":
    main()
