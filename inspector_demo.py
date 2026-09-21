"""Watch a round play out with a random policy.

    python inspector_demo.py                 # spectator window
    python inspector_demo.py --headless      # no window, prints the outcome

Use the arrow keys / mouse to fly the spectator camera.
"""

import argparse
import time

import numpy as np

from download_decompiled_map import download_map_if_not_exist
from env.csgo_environment import env


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--team-size", type=int, default=2, help="agents per team")
    p.add_argument("--episodes", type=int, default=1, help="rounds to play")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--headless", action="store_true", help="run without a window")
    p.add_argument("--show-waypoints", action="store_true")
    p.add_argument("--no-minimap", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    # The decompiled map is not tracked in git.
    download_map_if_not_exist()

    my_env = env(
        num_team_agents=args.team_size,
        render_mode=None if args.headless else "spectator",
        debug_mode=not args.headless,
        show_waypoints=args.show_waypoints,
        show_minimap=not (args.headless or args.no_minimap),
        seed=args.seed,
    )

    rng = np.random.default_rng(args.seed)

    for episode in range(args.episodes):
        my_env.reset(seed=args.seed + episode)
        start = time.time()
        steps = 0

        # agent_iter ends by itself once every agent has terminated and been
        # retired, so no manual "are they all done?" check is needed.
        for agent in my_env.agent_iter():
            observation, reward, termination, truncation, info = my_env.last()

            record = my_env.state_sequence[agent]
            record["observation"].append(observation)
            record["reward"].append(reward)
            record["termination"].append(termination)
            record["truncation"].append(truncation)
            record["episode_length"] += 1

            if termination or truncation:
                # A terminated agent must be stepped once with None to retire it.
                my_env.step(None)
                continue

            legal_actions = np.flatnonzero(info["action_mask"])
            my_env.step(int(rng.choice(legal_actions)))
            steps += 1

        winning_team, reason = my_env.unwrapped.round_outcome()
        elapsed = time.time() - start
        print(f"round {episode + 1}: {winning_team.name} win by {reason.name} "
              f"({steps} decisions, {elapsed:.1f}s, {steps/elapsed:.0f} steps/s)")

        for agent_id, stats in my_env.unwrapped.agent_stats().items():
            print(f"    {agent_id:<5} kills={stats['kills']} deaths={stats['deaths']} "
                  f"damage={stats['damage_dealt']:.0f} distance={stats['total_distance']:.0f}")

    my_env.close()


if __name__ == "__main__":
    main()
