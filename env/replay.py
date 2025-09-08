from csgo_environment import env
import numpy as np
from policy import ReplayPolicy
import os
from utils import transform_csgo_to_panda3d
from tqdm import tqdm 
from multiprocessing import Pool
from utils import Weapon
import signal

def should_trigger_timeout(player_policies, replay_env):
    """Check if all players have either completed their actions or died"""
    for agent_id, policy in player_policies.items():
        agent = replay_env.engine.agents[agent_id]
        # Continue if either the agent is dead or has completed their action sequence
        if not agent.is_alive or policy.current_index >= len(policy.action_sequence):
            continue
        return False
    return True

def process_replay_file(args):
    """Process a single replay file"""
    data_file, replay_folder = args
    replay_env = env(num_team_agents=5, render_mode=None, debug_mode=False)
    waypoint_graph = replay_env.engine.waypoints

    replay_data = np.load(os.path.join(replay_folder, data_file), allow_pickle=True)
    all_player_positions = replay_data["player_trajectory"]
    panda3d_positions = transform_csgo_to_panda3d(all_player_positions)

    player_policies = {}
    player_spawns = {}
    round_id = data_file.replace(".npz", "")

    for player_idx in range(len(panda3d_positions)):
        player_id = replay_data["player_ids"][player_idx]
        seq_len = replay_data["player_seq_len"][player_idx]
        # player_positions = panda3d_positions[player_idx]
        player_positions = panda3d_positions[player_idx][:seq_len]
        waypoint_path, action_sequence = waypoint_graph.get_interpolated_waypoint_path(player_positions, return_actions=True)
        init_waypoint_id = waypoint_path[0]
        player_policies[player_id] = ReplayPolicy(action_sequence)
        player_spawns[player_id] = {
            "init_waypoint_id": init_waypoint_id
        }
    
    init_bomb_carrier = replay_data["initial_bomb_carrier_player_idx"]
    if init_bomb_carrier.item() is not None:
        init_bomb_carrier_id = replay_data["player_ids"][init_bomb_carrier.item()]
        init_bomb_position = None
    else:
        init_bomb_position = replay_data["initial_bomb_location"]
        init_bomb_position = transform_csgo_to_panda3d(init_bomb_position)
        init_bomb_carrier_id = None
    
    reset_options = {
        "player_spawns": player_spawns,
        "init_bomb_carrier_id": init_bomb_carrier_id,
        "init_bomb_position": init_bomb_position,
        "round_id": round_id
    }

    replay_env.reset(options=reset_options)

    for agent in replay_env.agent_iter():
        observation, reward, termination, truncation, info = replay_env.last()

        replay_env.state_sequence[agent]["observation"].append(observation)
        replay_env.state_sequence[agent]["reward"].append(reward)
        replay_env.state_sequence[agent]["termination"].append(termination)
        replay_env.state_sequence[agent]["truncation"].append(truncation)
        replay_env.state_sequence[agent]["episode_length"] += 1

        if all(replay_env.terminations.values()) or all(replay_env.truncations.values()):
            break

        if termination or truncation:
            action = None
        else:
            action, replay_ends = player_policies[agent].get_action(observation, info["action_mask"])
            if replay_ends:
                replay_env.engine.set_agent_hp(agent, 0)
            # Check if all agents have either completed actions or died
            if should_trigger_timeout(player_policies, replay_env):
                replay_env.engine.trigger_timeout()
                # print("Timeout triggered")
        replay_env.step(action)
    
    replay_env.close()

def main():
    # NUM_WORKERS = 1
    NUM_WORKERS = 32
    NUM_AGENTS = 5
    RENDER_MODE = None
    TOTAL_REPLAYS = 1000
    # RENDER_MODE = "spectator"
    if RENDER_MODE == "spectator" and NUM_WORKERS > 1:
        raise ValueError("Cannot use spectator mode with multiple workers")
    REPLAY_DATA_FOLDER = os.path.join("data", "player_seq_allmap_de_dust2_npz")
    replay_data_files = [f for f in os.listdir(REPLAY_DATA_FOLDER) if f.endswith(".npz")]

    if NUM_WORKERS > 1:
        original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        pool = Pool(NUM_WORKERS)
        signal.signal(signal.SIGINT, original_sigint_handler)
        file_args = [(f, REPLAY_DATA_FOLDER) for f in replay_data_files[:TOTAL_REPLAYS]]
        try:
            # Use imap_unordered instead of imap for faster response to interruptions
            for _ in tqdm(pool.imap_unordered(process_replay_file, file_args), total=len(replay_data_files)):
                pass
        except KeyboardInterrupt:
            print("\nCtrl+C received, terminating workers...")
            pool.terminate()  # Forcefully stop workers
        finally:
            pool.join()  # Ensure cleanup
    else:
        replay_env = env(num_team_agents=NUM_AGENTS, render_mode=RENDER_MODE, debug_mode=True if RENDER_MODE == "spectator" else False)
        waypoint_graph = replay_env.engine.waypoints

        for data_file in tqdm(replay_data_files):
            replay_data = np.load(os.path.join(REPLAY_DATA_FOLDER, data_file), allow_pickle=True)
            all_player_positions = replay_data["player_trajectory"]
            panda3d_positions = transform_csgo_to_panda3d(all_player_positions)

            player_policies = {}
            player_spawns = {}
            player_weapons = {}
            player_armor = {}
            player_helmet = {}
            round_id = data_file.replace(".npz", "")

            for player_idx in range(len(panda3d_positions)):
                player_id = replay_data["player_ids"][player_idx]
                seq_len = replay_data["player_seq_len"][player_idx]
                # player_positions = panda3d_positions[player_idx
                player_weapons[player_id] = Weapon[replay_data["player_weapons"][player_idx].replace("-", "_").replace(" ", "_")]
                player_armor[player_id] = replay_data["player_armor"][player_idx] > 0
                player_helmet[player_id] = replay_data["player_helmet"][player_idx] > 0
                player_positions = panda3d_positions[player_idx][:seq_len]
                waypoint_path, action_sequence = waypoint_graph.get_interpolated_waypoint_path(player_positions, return_actions=True)
                init_waypoint_id = waypoint_path[0]
                player_policies[player_id] = ReplayPolicy(action_sequence)
                player_spawns[player_id] = {
                    "init_waypoint_id": init_waypoint_id
                }
        
            init_bomb_carrier = replay_data["initial_bomb_carrier_player_idx"]
            if init_bomb_carrier.item() is not None:
                init_bomb_carrier_id = replay_data["player_ids"][init_bomb_carrier.item()]
                init_bomb_position = None
            else:
                init_bomb_position = replay_data["initial_bomb_location"]
                init_bomb_position = transform_csgo_to_panda3d(init_bomb_position)
                init_bomb_carrier_id = None
        
            reset_options = {
                "player_spawns": player_spawns,
                "player_weapons": player_weapons,
                "player_armor": player_armor,
                "player_helmet": player_helmet,
                "init_bomb_carrier_id": init_bomb_carrier_id,
                "init_bomb_position": init_bomb_position,
                "round_id": round_id
            }

            replay_env.reset(options=reset_options)

            for agent in replay_env.agent_iter():
                observation, reward, termination, truncation, info = replay_env.last()

                replay_env.state_sequence[agent]["observation"].append(observation)
                replay_env.state_sequence[agent]["reward"].append(reward)
                replay_env.state_sequence[agent]["termination"].append(termination)
                replay_env.state_sequence[agent]["truncation"].append(truncation)
                replay_env.state_sequence[agent]["episode_length"] += 1

                if all(replay_env.terminations.values()) or all(replay_env.truncations.values()):
                    break

                if termination or truncation:
                    action = None
                else:
                    action, replay_ends = player_policies[agent].get_action(observation, info["action_mask"])
                    if replay_ends:
                        replay_env.engine.set_agent_hp(agent, 0)
                    # Check if all agents have either completed actions or died
                    if should_trigger_timeout(player_policies, replay_env):
                        replay_env.engine.trigger_timeout()
                        # print("Timeout triggered")
                replay_env.step(action)
        
        replay_env.close()


if __name__ == "__main__":
    main()