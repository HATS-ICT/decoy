import sys
import os
from collections import deque, defaultdict
import random
import uuid
from typing import Optional, Dict, Any, Literal, List

import numpy as np
from panda3d.core import loadPrcFileData, Vec3, DirectionalLight, AmbientLight, LineSegs
from panda3d.bullet import BulletWorld, BulletRigidBodyNode, BulletTriangleMeshShape, BulletTriangleMesh
from direct.showbase.ShowBase import ShowBase

from .config import *
from .agent import Agent
from .waypoints import WaypointGraph
from .utils import Direction, Team, BombStatus, WinReason, vec_distance
from .debug import DebugManager
from .camera import SpectatorCamera
from .damage_model import DamageEstimateModel

class GameStateLogger:
    
    """Tracks the history of game states throughout a round."""
    def __init__(self, round_id: int):
        self.round_id = round_id
        self.agent_hp_trajectories = defaultdict(list)
        self.agent_position_trajectories = defaultdict(list)
        self.bomb_position_trajectory = []
        self.bomb_status_trajectory = []
        self.winning_team = None
        self.winning_reason = None
        self.agent_sequence_lengths = defaultdict(int)
        
        # Store agent IDs in a consistent order
        self.agent_ids = [f"T_{i}" for i in range(5)] + [f"CT_{i}" for i in range(5)]

    def update(self, agents: Dict[str, Agent], bomb_position: Vec3, bomb_status: BombStatus):
        """Update all trajectories with current game state."""
        # Update agent trajectories
        for agent_id, agent in agents.items():
            self.agent_hp_trajectories[agent_id].append(agent.display_health)
            self.agent_position_trajectories[agent_id].append(
                [agent.position.x, agent.position.y, agent.position.z]
            )
            if agent.is_alive:
                self.agent_sequence_lengths[agent_id] += 1

        # Update bomb trajectories
        self.bomb_position_trajectory.append(
            [bomb_position.x, bomb_position.y, bomb_position.z]
        )
        self.bomb_status_trajectory.append(bomb_status)

    def set_outcome(self, winning_team: Team, winning_reason: WinReason):
        """Set the final outcome of the round."""
        self.winning_team = winning_team
        self.winning_reason = winning_reason

    def export_to_file(self, filepath: str):
        """Export the game log to a .npz file."""
        # Convert position trajectories to numpy array (num_agents, seq_len, 3)
        max_len = max(len(traj) for traj in self.agent_position_trajectories.values())
        player_trajectory = np.zeros((10, max_len, 3))
        
        for idx, agent_id in enumerate(self.agent_ids):
            traj = self.agent_position_trajectories[agent_id]
            player_trajectory[idx, :len(traj)] = np.array(traj)
            # Pad remaining timesteps with last position
            if len(traj) < max_len:
                player_trajectory[idx, len(traj):] = player_trajectory[idx, len(traj)-1]

        # Convert HP trajectories to numpy array (num_agents, seq_len)
        player_hp_timeseries = np.zeros((10, max_len))
        for idx, agent_id in enumerate(self.agent_ids):
            traj = self.agent_hp_trajectories[agent_id]
            player_hp_timeseries[idx, :len(traj)] = np.array(traj)
            # Pad remaining timesteps with last HP
            if len(traj) < max_len:
                player_hp_timeseries[idx, len(traj):] = player_hp_timeseries[idx, len(traj)-1]

        # Convert sequence lengths to numpy array
        player_seq_len = np.zeros(10, dtype=np.int32)
        for idx, agent_id in enumerate(self.agent_ids):
            player_seq_len[idx] = self.agent_sequence_lengths.get(agent_id, 0)

        # Convert bomb trajectory to numpy array
        bomb_trajectory = np.array(self.bomb_position_trajectory)

        # Save to npz file
        np.savez(
            filepath,
            player_trajectory=player_trajectory,
            player_hp_timeseries=player_hp_timeseries,
            player_ids=np.array(self.agent_ids),
            player_seq_len=player_seq_len,  # Add sequence lengths to saved data
            bomb_trajectory=bomb_trajectory,
            round_end_reason=np.array(self.winning_reason.name),
            winning_side=np.array(self.winning_team.name)
        )

class CSGOEngine(ShowBase):
    def __init__(self, 
                 num_team_agents: int,
                 render_mode: Optional[Literal["spectator"]] = None, 
                 debug_mode: bool = False,
                 show_waypoints: bool = False,
                 show_minimap: bool = False,
                 waypoint_data_path: Optional[str] = None):
        self._init_panda3d_settings(render_mode)
        super().__init__()
        
        # Core settings
        self.num_team_agents = num_team_agents
        self.render_mode = render_mode
        self.debug_mode = debug_mode and render_mode is not None
        self.show_waypoints = show_waypoints and render_mode is not None
        self.show_minimap = show_minimap and render_mode is not None

        self.enable_logging = ENABLE_LOGGING
        if self.enable_logging:
            os.makedirs(LOG_FOLDER, exist_ok=True)
        
        # Initialize simulation state
        self._init_game_state()
        self.logger = GameStateLogger(self.round_id)
        
        # Initialize core components
        # Create debug manager if any visual debug feature is enabled
        self.debug_manager = DebugManager(self) if (self.debug_mode or self.show_waypoints or self.show_minimap) else None
        self._init_map()
        self._init_physics()
        self._init_lighting()
        self._init_waypoint_system(waypoint_data_path)
        
        if ENABLE_AGENT_SHOOTING:
            self.damage_model = DamageEstimateModel()

        # Initialize debug and spectator features
        if render_mode == "spectator":
            self.spectator_camera = SpectatorCamera(self)
            # self.taskMgr.step() # stepping one frame to ensure rendering is ready
            self._init_camera()


    def _init_panda3d_settings(self, render_mode: Optional[str]) -> None:
        """Initialize Panda3D specific settings."""
        loadPrcFileData('', 'bullet-filter-algorithm groups-mask')
        loadPrcFileData("", "notify-level-util error")
        # Add window size configuration
        # loadPrcFileData('', 'win-size 1920 1080')  # Set to 1920x1080 resolution
        if render_mode is None:
            loadPrcFileData('', 'window-type none')
            loadPrcFileData('', 'audio-library-name null')

    def _init_game_state(self) -> None:
        """Initialize simulation state variables."""
        self.round_id = uuid.uuid4()
        self.time_scale = TIME_SCALE
        self.physics_step = PHYSICS_STEP
        self.render_frame_interval = RENDER_FRAME_INTERVAL
        self.simulation_paused = False
        self.single_step_requested = False
        self.physics_time_buffer = 0.0
        self.physics_time_cumulative = 0.0
        self.physics_ticks = 0
        self.time_since_render = 0.0
        self.total_agent_decision_requests = 0
        self.last_realtime = 0
        self.agents: Dict[str, Agent] = {}
        self.agents_by_team: Dict[Team, List[Agent]] = {Team.T: [], Team.CT: []}
        self.agent_action_request_queue = deque()
        self.termination_queue = deque()
        self.bomb_status = BombStatus.Dropped
        self.bomb_world_position = None
        self.bomb_carrier = None
        self.winning_team = None
        self.winning_reason = None
        self.game_time_limit = GAME_TIME_LIMIT
        self.game_timeout_flag = False

    def _init_map(self):
        """Initialize the map model and its properties."""
        try:
            self.map = self.loader.loadModel(MAP_PATH, noCache=NO_MODEL_CACHE)
        except Exception as e:
            print(f"Error loading FBX map: {e}")
            sys.exit(1)
        
        self.map.reparentTo(self.render)
        self.map.set_pos(MAP_POSITION)
        self.map.set_scale(MAP_SCALE)
        self.map.set_p(MAP_ROTATION[0])
        self.map.set_h(MAP_ROTATION[1])
        self.map.set_r(MAP_ROTATION[2])

    def _init_physics(self):
        """Initialize the physics world and its properties."""
        self.world = BulletWorld()
        self.world.setGravity(Vec3(0, 0, GRAVITY))

        # Create a triangle mesh for the map
        mesh = BulletTriangleMesh()
        for geom_node in self.map.findAllMatches('**/+GeomNode'):
            geom_node = geom_node.node()
            for geom in geom_node.getGeoms():
                mesh.addGeom(geom)
        
        shape = BulletTriangleMeshShape(mesh, dynamic=False)
        node = BulletRigidBodyNode('MapCollision')
        node.addShape(shape)
        # Add some friction to prevent sliding
        np = self.render.attachNewNode(node)
        np.setPos(self.map.getPos())
        np.setHpr(self.map.getHpr())
        np.setScale(self.map.getScale())
        np.setCollideMask(COLLISION_BITMASK_ENVIRONMENT)
        # node.setFriction(0.0)
        self.world.attachRigidBody(node)

        self.world.setGroupCollisionFlag(1, 1, False)
        self.world.setGroupCollisionFlag(0, 1, True)

    def _init_lighting(self):
        """Initialize the lighting setup."""
        # Create a directional light
        directional_light = DirectionalLight("directional_light")
        directional_light.setColor(DIRECTIONAL_LIGHT_COLOR)
        directional_light_np = self.render.attachNewNode(directional_light)
        directional_light_np.setPos(DIRECTIONAL_LIGHT_POS)
        directional_light_np.setHpr(DIRECTIONAL_LIGHT_ROT)
        self.render.setLight(directional_light_np)

        # Add ambient light
        ambient_light = AmbientLight("ambient_light")
        ambient_light.setColor(AMBIENT_LIGHT_COLOR)
        ambient_light_np = self.render.attachNewNode(ambient_light)
        self.render.setLight(ambient_light_np)

    def _init_camera(self):
        if self.render_mode is None:
            return 
        
        # Set fixed position and orientation
        self.cam.setPos(INITIAL_CAMERA_POS)
        self.cam.setHpr(INITIAL_CAMERA_ROT)  # Heading, Pitch, Roll
        self.camLens.setFov(CAMERA_FOV)

    def _init_waypoint_system(self, waypoint_data_path: Optional[str]) -> None:
        """Initialize the waypoint system."""
        self.waypoints = WaypointGraph()
        self.waypoint_data_path = waypoint_data_path or WAYPOINT_DATA_PATH
        self.waypoints.load_from_json(self.waypoint_data_path)
        
        if self.show_waypoints and self.debug_manager:
            self.debug_manager.create_waypoint_visualization()

    @property
    def game_time(self) -> float:
        """Current game time in seconds."""
        return self.physics_time_cumulative

    @property
    def time_remaining(self) -> float:
        """Time remaining in the game in seconds."""
        return self.game_time_limit - self.game_time

    @property
    def game_ended(self):
        return self.winning_team is not None

    @property
    def alive_agents(self):
        """Returns an iterator of agent IDs for all currently alive agents."""
        return (agent_id for agent_id, agent in self.agents.items() if agent.is_alive)

    @property
    def alive_t_agents(self):
        """Returns list of alive Terrorist agents."""
        return [agent for agent in self.agents_by_team[Team.T] if agent.is_alive]

    @property
    def alive_ct_agents(self):
        """Returns list of alive Counter-Terrorist agents."""
        return [agent for agent in self.agents_by_team[Team.CT] if agent.is_alive]

    @property
    def num_alive_t(self) -> int:
        """Number of alive Terrorist agents."""
        return len(self.alive_t_agents)

    @property
    def num_alive_ct(self) -> int:
        """Number of alive Counter-Terrorist agents."""
        return len(self.alive_ct_agents)
    
    @property
    def bomb_has_planted(self):
        return self.bomb_status == BombStatus.Planted
    
    @property
    def current_bomb_position(self):
        """
        Returns the current world position of the bomb, whether it's dropped, 
        planted, or being carried by an agent.
        """
        if self.bomb_world_position is not None:
            assert self.bomb_status in [BombStatus.Dropped, BombStatus.Planted, BombStatus.Detonated, BombStatus.Defused], f"Bomb is not dropped or planted when not having a world position: {self.bomb_status}"
            return self.bomb_world_position
        else:
            assert self.bomb_carrier is not None, f"Bomb is not being carried when not having a world position {self.bomb_status}"
            return self.agents[self.bomb_carrier].position
    
    def update_simulation(self):
        """Update the simulation state by advancing physics and handling agent updates.
        Continues updating until an agent requests a decision."""
        while not self.agent_action_request_queue:
            # Update time tracking
            realtime = self.clock.getRealTime()
            realtime_dt = realtime - self.last_realtime
            self.last_realtime = realtime
            
            # Accumulate time for physics and rendering
            self.physics_time_buffer += realtime_dt * self.time_scale
            self.time_since_render += realtime_dt
            # Process physics steps
            while self.should_process_physics():
                
                self.progress_game()
                self.process_physics_step()

                if self.should_log_game_state():
                    self.logger.update(self.agents, self.current_bomb_position, self.bomb_status)
                
                # Handle rendering if needed
                if self.should_render():
                    self.handle_rendering()
                
                self.physics_time_buffer -= self.physics_step
                
                # Check if any agent needs a decision
                if self.game_ended or self.agent_decision_request_check():
                    break
            if self.game_ended:
                if self.enable_logging:
                    self.logger.set_outcome(self.winning_team, self.winning_reason)
                    self.logger.export_to_file(os.path.join(LOG_FOLDER, f"{self.round_id}.npz"))
                break

    def progress_game(self):
        """Update the game state."""
        # Update agent movements - only for alive agents
        for agent_id in self.alive_agents:
            self.agents[agent_id].update_movement()

        # handle shooting & grenade - only for alive agents
        for agent_id in self.alive_agents:
            self.agents[agent_id].handle_automatic_game_actions()

        if self.should_model_damage_events():
            self.estimate_damage_outcomes()

        # handle agent death - only for alive agents
        for agent_id in self.alive_agents:
            self.agents[agent_id].handle_death()

        # handle bomb planting and defusing
        for agent_id in self.alive_agents:
            self.agents[agent_id].handle_bomb_actions()

        self.update_game_state()

        # check for game end
        self.check_winning_team()
        if self.game_ended:
            for agent_id in self.alive_agents:
                self.termination_queue.append(agent_id)

    def process_physics_step(self):
        """Process a single physics step."""
        if not self.simulation_paused or self.single_step_requested:
            self.world.doPhysics(self.physics_step)
            self.physics_ticks += 1
            self.physics_time_cumulative += self.physics_step
            
            if self.single_step_requested:
                self.single_step_requested = False

    def should_process_physics(self):
        return self.render_mode is None or self.physics_time_buffer > self.physics_step
    
    def should_log_game_state(self):
        return self.enable_logging and self.game_time % LOG_FREQUENCY < self.physics_step
    
    def should_model_damage_events(self):
        return ENABLE_AGENT_SHOOTING and self.game_time % DAMAGE_MODEL_FREQUENCY < self.physics_step

    def should_render(self):
        """Determine if rendering should occur."""
        return self.render_mode is not None and self.time_since_render >= self.render_frame_interval

    def handle_rendering(self):
        """Handle rendering and debug visualization."""
        if self.debug_manager:
            if self.debug_mode:
                for agent_id in self.alive_agents:
                    self.debug_manager.update_agent_debug_line(agent_id)
            for line in self.debug_manager.shooting_lines:
                line.removeNode()
            self.debug_manager.shooting_lines = []
        self.taskMgr.step()
        self.time_since_render = 0.0

    def update_game_state(self):
        """Update the game state."""
        if self.time_remaining <= 0:
            if self.bomb_has_planted:
                self.bomb_status = BombStatus.Detonated
            self.game_timeout_flag = True

    def estimate_damage_outcomes(self):
        """Estimate the damage outcomes for all alive agents."""
        # Get all alive agents from both teams
        alive_t_agents = self.alive_t_agents
        alive_ct_agents = self.alive_ct_agents
        
        # Check damage between pairs of agents both ways
        for t_agent in alive_t_agents:
            for ct_agent in alive_ct_agents:
                if t_agent.has_line_of_sight_to_agent(ct_agent):
                    self.estimate_damage(attacker=t_agent, victim=ct_agent)
                if ct_agent.has_line_of_sight_to_agent(t_agent):
                    self.estimate_damage(attacker=ct_agent, victim=t_agent)

    def estimate_damage(self, attacker, victim):
        """Estimate the damage outcome for a given attacker and victim."""
        will_damage, damage_amount, hit_group = self.damage_model.predict_damage(attacker.position, victim.position, attacker.view_angle, victim.view_angle, attacker.health, attacker.weapon.value, victim.has_armor, victim.has_helmet)
        if will_damage:
            victim.health -= damage_amount
            attacker.draw_shooting_line(victim)
            # print(f"Damage estimated: {damage_amount} to {victim.agent_id} from {attacker.agent_id}")
        return 

    def check_winning_team(self):
        """Check if the game has ended. return winning team and reason"""
        if self.bomb_status == BombStatus.Detonated:
            self.winning_team, self.winning_reason = Team.T, WinReason.BombDetonated
        elif self.bomb_status == BombStatus.Defused:
            self.winning_team, self.winning_reason = Team.CT, WinReason.BombDefused
        elif self.num_alive_t == 0:
            self.winning_team, self.winning_reason = Team.CT, WinReason.TerroristEliminated
        elif self.num_alive_ct == 0:
            self.winning_team, self.winning_reason = Team.T, WinReason.CounterTerroristEliminated
        elif self.game_timeout_flag: 
            self.winning_team, self.winning_reason = Team.CT, WinReason.TimeOut
        else:
            self.winning_team, self.winning_reason = None, None
        # if self.winning_team is not None:
        #     print(f"Game ended. Winning team: {self.winning_team}, Reason: {self.winning_reason}")

    def add_agent(self, agent_id, spawn_mode="random_spawn", init_pos=None, init_waypoint_id=None, weapon=None, has_armor=False, has_helmet=False):
        """Add a new agent to the game world."""
        assert spawn_mode in ["random", "random_spawn", "fixed"], "Invalid spawn mode"
        if spawn_mode == "fixed":
            assert (init_pos is not None) != (init_waypoint_id is not None), \
                "Either initial_pos or init_waypoint must be provided for fixed spawn mode (but not both)"
        team = Team[agent_id.split("_")[0]]
        agent = Agent(self, agent_id, team, spawn_mode, init_pos, init_waypoint_id, weapon, has_armor, has_helmet)
        self.agents[agent_id] = agent
        self.agents_by_team[team].append(agent)

        if self.debug_manager:
            ls = LineSegs()
            ls.setThickness(3.0)
            ls.setColor(1, 0, 0, 1)
            debug_node = ls.create()
            self.debug_manager.debug_lines[agent_id] = self.render.attachNewNode(debug_node)

    def set_move_target(self, agent_id, action):
        """Set the target position for an agent based on the given action."""
        agent = self.agents[agent_id]
        if action == 8: # stop
            agent.set_stop_action()
        else:
            direction = Direction(action)
            neighbor_id = agent.current_waypoint["neighbor_ids"][direction]
            target_waypoint = self.waypoints.get_waypoint_by_id(neighbor_id)
            agent.set_move_target(direction, target_waypoint)

    def trigger_timeout(self):
        self.game_timeout_flag = True

    def get_agent_state(self, agent_id):
        """Get the current state of an agent."""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        observation = self.agents[agent_id].observation
        reward = self.agents[agent_id].reward
        termination = self.agents[agent_id].termination
        truncation = False
        action_mask = self.agents[agent_id].action_mask
        info = {"action_mask": action_mask}
        return observation, reward, termination, truncation, info
    
    def get_next_agent(self):
        """
        Get the next agent that can take an action.
        If death occurs, it is immediately observed.
        """
        if self.termination_queue:
            return self.termination_queue.popleft()
        return self.agent_action_request_queue.popleft()
    
    def get_random_agent(self, team: Team, alive_only: bool = True, return_id: bool = True):
        """Get a random agent from the given team."""
        agents = self.agents_by_team[team]
        if alive_only:
            agents = [agent for agent in agents if agent.is_alive]
        if return_id:
            return random.choice(agents).agent_id
        else:
            return random.choice(agents)

    def agent_decision_request_check(self):
        """Check if an agent has requested a decision."""
        has_decision_request = False
        for agent_id in self.alive_agents:
            if self.agents[agent_id].has_decision_request():
                self.agent_action_request_queue.append(agent_id)
                self.total_agent_decision_requests += 1
                has_decision_request = True
        return has_decision_request
    
    def plant_bomb(self, plant_waypoint_id):
        self.bomb_world_position = self.waypoints.get_waypoint_by_id(plant_waypoint_id, return_pos=True)
        self.bomb_status = BombStatus.Planted
        self.bomb_carrier = None
        self.game_time_limit = self.game_time + GAME_TIME_BOMB_EXTENSION

    def defuse_bomb(self):
        self.bomb_status = BombStatus.Defused

    def estimate_hit_damage(self, attacker, victim):
        DISTANCE_MEAN = 705.522216796875
        DISTANCE_STD = 383.8244934082031
        weapon = attacker.weapon
        distance = vec_distance(attacker.position, victim.position) * COORDINATE_SCALE
        distance = (distance - DISTANCE_MEAN) / DISTANCE_STD
        has_armor = victim.has_armor
        has_helmet = victim.has_helmet
        damage = self.damage_model.predict_damage(weapon, distance, has_armor, has_helmet)
        return damage
    
    def set_agent_hp(self, agent_id, hp):
        self.agents[agent_id].health = hp

    def reset(self, options: Optional[Dict[str, Any]] = None):
        """Reset the game engine."""
        options = options or {}
        self.round_id = options.get("round_id", uuid.uuid4()) 
        self.logger = GameStateLogger(self.round_id)
        self.simulation_paused = False
        self.single_step_requested = False
        self.physics_time_buffer = 0.0
        self.physics_time_cumulative = 0.0
        self.physics_ticks = 0
        self.time_since_render = 0.0
        self.total_agent_decision_requests = 0
        self.last_realtime = 0
        self.agent_action_request_queue.clear()
        self.termination_queue.clear()
        self.clock.reset()
        self.bomb_status = BombStatus.Dropped
        self.bomb_world_position = None
        self.bomb_carrier = None
        self.winning_team = None
        self.winning_reason = None
        self.game_timeout_flag = False
        if self.debug_manager:
            self.debug_manager.reset()
        
        for agent_id in self.alive_agents:
            self.agents[agent_id].remove_model_assets()

        self.agents.clear()
        self.agents_by_team = {Team.T: [], Team.CT: []}  

        spawn_options = options.get("player_spawns", {})
        player_weapons = options.get("player_weapons", {})
        player_armor = options.get("player_armor", {})
        player_helmet = options.get("player_helmet", {})
        for i in range(self.num_team_agents):
            for t in ["T", "CT"]:
                agent_id = f"{t}_{i}"
                agent_options = spawn_options.get(agent_id, {})
                
                spawn_mode = "fixed" if agent_options else "random_spawn"
                init_pos = agent_options.get("init_pos", None)
                init_waypoint_id = agent_options.get("init_waypoint_id", None)
                weapon = player_weapons.get(agent_id, None)
                has_armor = player_armor.get(agent_id, False)
                has_helmet = player_helmet.get(agent_id, False)
                self.add_agent(agent_id, spawn_mode, init_pos, init_waypoint_id, weapon, has_armor, has_helmet)
                
        init_bomb_carrier_id = options.get("init_bomb_carrier_id", None)
        init_bomb_position = options.get("init_bomb_position", None)
        if init_bomb_carrier_id is not None:
            self.bomb_status = BombStatus.Carried
            self.bomb_carrier = init_bomb_carrier_id
        elif init_bomb_position is not None:
            self.bomb_status = BombStatus.Dropped
            self.bomb_world_position = Vec3(*init_bomb_position)
        else:
            self.bomb_status = BombStatus.Carried
            self.bomb_carrier = self.get_random_agent(Team.T, return_id=True)
        
        for agent_id in self.agents:
            self.agents[agent_id].reset()
            self.agent_action_request_queue.append(agent_id)

        if self.enable_logging:
            self.logger.update(self.agents, self.current_bomb_position, self.bomb_status)


if __name__ == "__main__":
    game = CSGOEngine()
    while True:
        game.step()