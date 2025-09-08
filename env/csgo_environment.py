import time
import functools

from gymnasium.spaces import Discrete, Box
import numpy as np

from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from panda3d.core import Vec3

from .game_engine import CSGOEngine
from .config import COORDINATE_SCALE


def env(num_team_agents=5, render_mode=None, debug_mode=False, show_waypoints=False, show_minimap=False):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    env = raw_env(num_team_agents, render_mode=render_mode, debug_mode=debug_mode, 
                  show_waypoints=show_waypoints, show_minimap=show_minimap)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    return env


class raw_env(AECEnv):
    metadata = {"render_modes": ["human"], "name": "csgo"}

    def __init__(self, num_team_agents, render_mode=None, debug_mode=False, show_waypoints=False, show_minimap=False):
        super().__init__()
        self.possible_agents = [f"{team}_{i}" for team in ["T", "CT"] for i in range(num_team_agents)]
        self.agents = self.possible_agents[:]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        self.render_mode = render_mode
        self.engine = CSGOEngine(num_team_agents, render_mode=render_mode, debug_mode=debug_mode,
                                show_waypoints=show_waypoints, show_minimap=show_minimap)

        self.state_sequence = {
            agent: {
                "observation": [],
                "reward": [],
                "termination": [],
                "truncation": [],
                "episode_length": 0
            }
            for agent in self.agents
        }

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # Increased to 6 dimensions to include target position (x,y,z, target_x, target_y, target_z)
        return Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(9)
    
    def observe(self, agent):
        observation, reward, termination, truncation, info = self.engine.get_agent_state(agent)
        self.rewards[agent] = reward
        self.terminations[agent] = termination
        self.truncations[agent] = truncation
        self.infos[agent]["action_mask"] = info["action_mask"]
        return observation
    
    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - terminations
        - truncations
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.engine.reset(options)
        self.agent_selection = self.engine.get_next_agent()

        self.state_sequence = {
            agent: {
                "observation": [],
                "reward": [],
                "termination": [],
                "truncation": [],
                "episode_length": 0
            }
            for agent in self.agents
        }

    def step(self, action):
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - terminations
        - truncations
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """
        agent = self.agent_selection

        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        self._cumulative_rewards[agent] = 0

        if action is not None:
            self.engine.set_move_target(agent, action)
            self.engine.update_simulation() # result is either game ended or until next agent decision request

        self.agent_selection = self.engine.get_next_agent()

    def close(self):
        self.engine.destroy()

def transform_csgo_to_panda3d(x, y, z, return_vec3=False):
    """
    Transforms a point from replay data space directly to Panda3D space.
    
    The transformation pipeline is as follows:
      1. The replay data point is first interpreted as (x, y, -z)
         (this is how the JSON data is loaded into Unity).
      2. It is then scaled by a uniform factor.
      3. A rotation given by Euler angles (90°, 270°, 90°) is applied.
      4. A translation is added.
      5. Finally, the Unity coordinate is converted to Panda3D space via:
            Panda3D_x = -Unity_x,
            Panda3D_y = -Unity_z,
            Panda3D_z =  Unity_y.
    
    Parameters:
        x, y, z (float): The coordinates from the replay JSON file.
    
    Returns:
        np.array: The transformed (x, y, z) coordinate in Panda3D space.
    """
    # --- Step 1: Convert replay data to initial Unity coordinate ---
    # In the replay code, the data is loaded as:
    #     Vector3(position[0], position[1], -position[2])
    point = np.array([x, y, z])
    
    # --- Step 2: Apply scaling ---
    # Unity scaling factor from the GameReplayManager (0.017 on each axis)
    # scaling = np.array([0.017, 0.017, 0.017])
    scaling = np.array([COORDINATE_SCALE, COORDINATE_SCALE, COORDINATE_SCALE])
    point = point * scaling
    translation = np.array([0.0, 0.0, 3.1])
    point = point + translation

    if return_vec3:
        return Vec3(point[0], point[1], point[2])
    else:
        return point
    

def get_a_panda3d_position_sequence(file_idx=10):
    REPLAY_DATA_FOLDER = os.path.join("data", "player_seq_allmap_de_dust2_npz")
    replay_data_files = [f for f in os.listdir(REPLAY_DATA_FOLDER) if f.endswith(".npz")]
    data_file = replay_data_files[file_idx]
    replay_data = np.load(os.path.join(REPLAY_DATA_FOLDER, data_file))
    all_player_positions = replay_data["player_trajectory"]
    return transform_csgo_to_panda3d(all_player_positions)
    

if __name__ == "__main__":
    my_env = env(num_team_agents=2, render_mode="spectator", debug_mode=True, show_waypoints=False, show_minimap=True)
    # my_env = env(num_team_agents=2, render_mode=None, debug_mode=False)
    my_env.reset()
    step_count = 0
    start_time = time.time()
    last_print_time = start_time

    # panda3d_positions = get_a_panda3d_position_sequence(file_idx=10)
    # for positions in panda3d_positions:
    #     my_env.env.engine.debug_manager.draw_connected_debug_lines(positions, color=(1, 0, 0, 1), thickness=3.0)
    
    MAX_STEPS = 10000
    for agent in my_env.agent_iter():
        observation, reward, termination, truncation, info = my_env.last()

        my_env.state_sequence[agent]["observation"].append(observation)
        my_env.state_sequence[agent]["reward"].append(reward)
        my_env.state_sequence[agent]["termination"].append(termination)
        my_env.state_sequence[agent]["truncation"].append(truncation)
        my_env.state_sequence[agent]["episode_length"] += 1
        # for agent in my_env.possible_agents:
        #     print(f"Agent {agent} episode length: {my_env.state_sequence[agent]['episode_length']}")

        if all(my_env.terminations.values()) or all(my_env.truncations.values()):
            time1 = time.time()
            # my_env.reset(options=test_spawn)
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


# if __name__ == "__main__":
#     my_env = env(num_team_agents=10, render_mode="spectator", debug_mode=True)
#     my_env.reset()
#     step_count = 0
#     start_time = time.time()
#     last_print_time = start_time
#     agent_actions = {agent: True for agent in my_env.possible_agents}  # Track each agent's action state
    
#     for agent in my_env.agent_iter():
#         print(my_env.engine.agent_action_request_queue)
#         observation, reward, termination, truncation, info = my_env.last()
#         # print(f"Agent {agent} observation: {observation}")
        
#         if termination or truncation:
#             action = None
#         else:
#             action = 0 if agent_actions[agent] else 2
#             agent_actions[agent] = not agent_actions[agent]  # Toggle this agent's next action
            
#         my_env.step(action)
#         step_count += 1
    
#     my_env.close()