from env.csgo_environment import env
from download_decompiled_map import download_map_if_not_exist
import time
import numpy as np


if __name__ == "__main__":
    # Ensure the map file is downloaded before starting
    download_map_if_not_exist()
    my_env = env(num_team_agents=2, render_mode="spectator", debug_mode=True, show_waypoints=False, show_minimap=True)
    # my_env = env(num_team_agents=2, render_mode=None, debug_mode=False)
    my_env.reset()
    step_count = 0
    start_time = time.time()
    last_print_time = start_time

    MAX_STEPS = 10000
    for agent in my_env.agent_iter():
        observation, reward, termination, truncation, info = my_env.last()

        my_env.state_sequence[agent]["observation"].append(observation)
        my_env.state_sequence[agent]["reward"].append(reward)
        my_env.state_sequence[agent]["termination"].append(termination)
        my_env.state_sequence[agent]["truncation"].append(truncation)
        my_env.state_sequence[agent]["episode_length"] += 1

        if all(my_env.terminations.values()) or all(my_env.truncations.values()):
            time1 = time.time()
            my_env.reset()
            continue

        if termination or truncation:
            action = None
        else:
            action_mask = info["action_mask"]
            valid_actions = np.where(action_mask)[0]
            action = np.random.choice(valid_actions)

        my_env.step(action)
        step_count += 1

        if step_count > MAX_STEPS:
            print("Max steps reached")
            break

    my_env.close()