import sys
import os
from collections import deque, defaultdict
import random
import uuid
from typing import Optional, Dict, Any, Literal, List

import numpy as np
import torch
from panda3d.core import loadPrcFileData, Vec3, DirectionalLight, AmbientLight, LineSegs
from panda3d.bullet import BulletWorld, BulletRigidBodyNode, BulletTriangleMeshShape, BulletTriangleMesh
from direct.showbase.ShowBase import ShowBase

from .config import *
from .agent import Agent
from .waypoints import WaypointGraph
from .utils import Direction, Team, BombStatus, WinReason, vec_distance, get_opposite_team, Region
from .debug import DebugManager
from .camera import SpectatorCamera
from .damage_model import DamageEstimateModel
from .rewards import RewardConfig, DEFAULT_REWARD_CONFIG

class GameStateLogger:
    
    """Tracks the history of game states throughout a round."""
    def __init__(self, round_id: int, agent_ids: Optional[List[str]] = None):
        self.round_id = round_id
        self.agent_hp_trajectories = defaultdict(list)
        self.agent_position_trajectories = defaultdict(list)
        self.bomb_position_trajectory = []
        self.bomb_status_trajectory = []
        self.winning_team = None
        self.winning_reason = None
        self.agent_sequence_lengths = defaultdict(int)
        
        # Store agent IDs in a consistent order. This used to be hardcoded to a
        # 5v5 roster, which silently fabricated rows for the absent agents (and
        # padded them from index -1) whenever the environment ran with a
        # different team size.
        if agent_ids is None:
            agent_ids = [f"T_{i}" for i in range(5)] + [f"CT_{i}" for i in range(5)]
        self.agent_ids = list(agent_ids)

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
        num_agents = len(self.agent_ids)
        recorded = [self.agent_position_trajectories.get(a, []) for a in self.agent_ids]
        max_len = max((len(t) for t in recorded), default=0)
        if max_len == 0:
            raise ValueError(f"Refusing to export round {self.round_id}: no states were recorded")

        # Convert position trajectories to numpy array (num_agents, seq_len, 3)
        player_trajectory = np.zeros((num_agents, max_len, 3))
        for idx, agent_id in enumerate(self.agent_ids):
            traj = self.agent_position_trajectories.get(agent_id, [])
            if not traj:
                continue
            player_trajectory[idx, :len(traj)] = np.array(traj)
            # Pad remaining timesteps with last position
            if len(traj) < max_len:
                player_trajectory[idx, len(traj):] = player_trajectory[idx, len(traj)-1]

        # Convert HP trajectories to numpy array (num_agents, seq_len)
        player_hp_timeseries = np.zeros((num_agents, max_len))
        for idx, agent_id in enumerate(self.agent_ids):
            traj = self.agent_hp_trajectories.get(agent_id, [])
            if not traj:
                continue
            player_hp_timeseries[idx, :len(traj)] = np.array(traj)
            # Pad remaining timesteps with last HP
            if len(traj) < max_len:
                player_hp_timeseries[idx, len(traj):] = player_hp_timeseries[idx, len(traj)-1]

        # Convert sequence lengths to numpy array
        player_seq_len = np.zeros(num_agents, dtype=np.int32)
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
                 waypoint_data_path: Optional[str] = None,
                 reward_config: Optional[RewardConfig] = None,
                 seed: Optional[int] = None):
        self._assert_no_live_engine()
        self._init_panda3d_settings(render_mode)
        super().__init__()
        
        # Core settings
        self.num_team_agents = num_team_agents
        self.reward_config = reward_config if reward_config is not None else DEFAULT_REWARD_CONFIG
        # Dedicated RNG streams so seeding an environment cannot be perturbed by,
        # or perturb, anything else using the global `random` / numpy state.
        self.rng = random.Random()
        self.np_rng = np.random.default_rng()
        self.seed(seed)
        self.render_mode = render_mode
        self.debug_mode = debug_mode and render_mode is not None
        self.show_waypoints = show_waypoints and render_mode is not None
        self.show_minimap = show_minimap and render_mode is not None

        self.enable_logging = ENABLE_LOGGING
        if self.enable_logging:
            os.makedirs(LOG_FOLDER, exist_ok=True)
        
        # Initialize simulation state
        self._init_game_state()
        self.logger = GameStateLogger(self.round_id, self.expected_agent_ids)
        
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


    @staticmethod
    def _assert_no_live_engine() -> None:
        """Fail early and legibly if an engine is already live in this process.

        CSGOEngine subclasses Panda3D's ShowBase, and Panda3D permits exactly
        one per process. Constructing a second raises a bare "Attempt to spawn
        multiple ShowBase instances!" from deep inside Panda3D, which is hard to
        connect back to the cause.

        Practical consequence: environments cannot be vectorised in-process.
        Run each one in its own subprocess, or close() the current environment
        before constructing the next.
        """
        import builtins
        if hasattr(builtins, "base"):
            raise RuntimeError(
                "A CSGOEngine is already running in this process. Panda3D allows "
                "only one ShowBase instance, so environments cannot coexist. "
                "Either call close() on the existing environment first, or run "
                "each environment in its own subprocess (see marl/vec_env.py)."
            )

    def _init_panda3d_settings(self, render_mode: Optional[str]) -> None:
        """Initialize Panda3D specific settings."""
        loadPrcFileData('', 'bullet-filter-algorithm groups-mask')
        loadPrcFileData("", "notify-level-util error")
        # Add window size configuration
        # loadPrcFileData('', 'win-size 1920 1080')  # Set to 1920x1080 resolution
        if render_mode is None:
            loadPrcFileData('', 'window-type none')
            loadPrcFileData('', 'audio-library-name null')

    @property
    def expected_agent_ids(self) -> List[str]:
        """Canonical agent ordering for this roster size."""
        return ([f"T_{i}" for i in range(self.num_team_agents)]
                + [f"CT_{i}" for i in range(self.num_team_agents)])

    def seed(self, seed: Optional[int] = None) -> None:
        """Reseed every stochastic component the simulation draws from.

        Covers spawn selection, bomb-carrier assignment and the torch generator
        used by the damage VAE, so a given seed reproduces a round exactly.
        """
        self.rng.seed(seed)
        self.np_rng = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)

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
        # check_winning_team() runs every tick; this latch keeps the terminal
        # reward from being granted more than once per round.
        self._terminal_reward_granted = False
        # Objective distance fields for reward shaping. The bomb-site field is
        # map-static and survives resets; the bomb field is rebuilt on each plant.
        self._bombsite_field = None
        self._bomb_field = None

    def _init_map(self):
        """Initialize the map model and its properties."""
        try:
            self.map = self.loader.loadModel(panda_path(MAP_PATH), noCache=NO_MODEL_CACHE)
        except Exception as e:
            raise FileNotFoundError(
                f"Could not load the map model at {MAP_PATH}: {e}\n"
                "The decompiled map is not tracked in git. Fetch it with:\n"
                "    python download_decompiled_map.py"
            ) from e
        
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
        # `np` here used to shadow the numpy import for the rest of the method.
        map_node_path = self.render.attachNewNode(node)
        map_node_path.setPos(self.map.getPos())
        map_node_path.setHpr(self.map.getHpr())
        map_node_path.setScale(self.map.getScale())
        map_node_path.setCollideMask(COLLISION_BITMASK_ENVIRONMENT)
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
        """Advance physics until an agent requests a decision, or the round ends."""
        if self.game_ended:
            # Re-entering after the round is decided would re-append every
            # surviving agent to the termination queue on each tick.
            return
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

    # ------------------------------------------------------------ objectives
    @property
    def bombsite_distance_field(self):
        """Distance from every waypoint to the nearest bomb site, computed once."""
        if self._bombsite_field is None:
            sites = (self.waypoints.waypoints_in_region(Region.A_BOMBSITE)
                     + self.waypoints.waypoints_in_region(Region.B_BOMBSITE))
            self._bombsite_field = self.waypoints.distance_field(sites)
        return self._bombsite_field

    def objective_distance(self, agent) -> Optional[float]:
        """Graph distance from ``agent`` to whatever it should be heading for.

        Before the plant both sides converge on the bomb sites; afterwards both
        converge on the bomb itself (CT to defuse it, T to hold it).
        """
        field = self._bomb_field if self.bomb_has_planted else self.bombsite_distance_field
        if field is None:
            return None
        waypoint = agent.current_waypoint
        if waypoint is None:
            return None
        return field.get(waypoint["id"])

    def credit_objective_progress(self, agent) -> None:
        """Reward the graph distance closed on the objective since the last decision."""
        weight = self.reward_config.objective_progress
        if weight == 0.0:
            return

        distance = self.objective_distance(agent)
        if distance is None:
            return

        previous = agent.prev_objective_distance
        agent.prev_objective_distance = distance
        if previous is None:
            # First decision of the round establishes the baseline only.
            return
        self.add_reward(agent.agent_id, weight * (previous - distance), share_with_team=False)

    def add_reward(self, agent_id: str, amount: float, share_with_team: bool = True) -> None:
        """Credit ``amount`` to ``agent_id``, optionally sharing with teammates.

        Rewards accumulate on the agent until the environment drains them in
        ``step()``; that is what makes the AEC reward semantics ("reward earned
        since this agent last acted") come out right.
        """
        if amount == 0.0:
            return
        self.agents[agent_id].pending_reward += amount

        spirit = self.reward_config.team_spirit
        if share_with_team and spirit > 0.0:
            team = self.agents[agent_id].team
            for teammate in self.agents_by_team[team]:
                if teammate.agent_id != agent_id:
                    teammate.pending_reward += amount * spirit

    def add_team_reward(self, team: Team, amount: float) -> None:
        """Credit ``amount`` to every agent on ``team``, alive or dead."""
        if amount == 0.0:
            return
        for agent in self.agents_by_team[team]:
            agent.pending_reward += amount

    def collect_rewards(self) -> Dict[str, float]:
        """Drain and return the reward each agent accrued since the last call."""
        drained = {}
        for agent_id, agent in self.agents.items():
            drained[agent_id] = agent.pending_reward
            agent.pending_reward = 0.0
        return drained

    def estimate_damage_outcomes(self):
        """Estimate the damage outcomes for all alive agents."""
        # Get all alive agents from both teams
        alive_t_agents = self.alive_t_agents
        alive_ct_agents = self.alive_ct_agents
        
        # Line of sight is symmetric, so it is tested once per unordered pair
        # rather than once per ordered pair; both firing directions are then
        # resolved from that single ray test.
        engagements = []
        for t_agent in alive_t_agents:
            for ct_agent in alive_ct_agents:
                if t_agent.has_line_of_sight_to_agent(ct_agent):
                    engagements.append((t_agent, ct_agent))
                    engagements.append((ct_agent, t_agent))

        if engagements:
            self.resolve_engagements(engagements)

    def resolve_engagements(self, engagements):
        """Resolve every (attacker, victim) pair with a single batched model call."""
        outcomes = self.damage_model.predict_damage_batch([
            dict(attacker_pos=a.position, victim_pos=v.position,
                 attacker_angle=a.view_angle, victim_angle=v.view_angle,
                 attacker_hp=a.health, attacker_weapon_id=a.weapon.value,
                 victim_has_armor=v.has_armor, victim_has_helmet=v.has_helmet)
            for a, v in engagements
        ])

        for (attacker, victim), (will_damage, damage_amount, _hit_group) in zip(engagements, outcomes):
            if will_damage:
                self.apply_damage(attacker, victim, damage_amount)

    def apply_damage(self, attacker, victim, damage_amount: float) -> None:
        """Apply damage, record attribution, and credit the shaping rewards.

        Attribution matters: without recording who landed the blow there is no
        way to credit the kill when the victim's health later reaches zero.
        """
        # Only damage that lands on a living agent counts, so overkill past
        # 0 HP is not rewarded.
        effective = min(damage_amount, max(victim.health, 0.0))
        victim.health -= damage_amount
        victim.last_attacker_id = attacker.agent_id

        cfg = self.reward_config
        self.add_reward(attacker.agent_id, cfg.damage_dealt * effective)
        self.add_reward(victim.agent_id, cfg.damage_taken * effective)

        attacker.stats.damage_dealt += effective
        victim.stats.damage_taken += effective
        attacker.draw_shooting_line(victim)

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

        if self.winning_team is not None and not self._terminal_reward_granted:
            self._terminal_reward_granted = True
            self.add_team_reward(self.winning_team, self.reward_config.win)
            self.add_team_reward(get_opposite_team(self.winning_team), self.reward_config.lose)
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
        if self.agent_action_request_queue:
            return self.agent_action_request_queue.popleft()
        return None
    
    def get_random_agent(self, team: Team, alive_only: bool = True, return_id: bool = True):
        """Get a random agent from the given team."""
        agents = self.agents_by_team[team]
        if alive_only:
            agents = [agent for agent in agents if agent.is_alive]
        if return_id:
            return self.rng.choice(agents).agent_id
        else:
            return self.rng.choice(agents)

    def agent_decision_request_check(self):
        """Check if an agent has requested a decision."""
        has_decision_request = False
        for agent_id in list(self.alive_agents):
            if self.agents[agent_id].has_decision_request():
                self.agent_action_request_queue.append(agent_id)
                self.total_agent_decision_requests += 1
                self.add_reward(agent_id, self.reward_config.time_penalty, share_with_team=False)
                self.credit_objective_progress(self.agents[agent_id])
                has_decision_request = True
        return has_decision_request
    
    def plant_bomb(self, plant_waypoint_id, planter_id: Optional[str] = None):
        self.bomb_world_position = self.waypoints.get_waypoint_by_id(plant_waypoint_id, return_pos=True)
        self.bomb_status = BombStatus.Planted
        self.bomb_carrier = None
        self.game_time_limit = self.game_time + GAME_TIME_BOMB_EXTENSION
        if self.reward_config.objective_progress != 0.0:
            self._bomb_field = self.waypoints.distance_field([plant_waypoint_id])
            # The objective moved, so every agent's baseline is stale.
            for agent in self.agents.values():
                agent.prev_objective_distance = None
        if planter_id is not None:
            self.add_reward(planter_id, self.reward_config.bomb_plant)

    def defuse_bomb(self, defuser_id: Optional[str] = None):
        self.bomb_status = BombStatus.Defused
        if defuser_id is not None:
            self.add_reward(defuser_id, self.reward_config.bomb_defuse)

    def set_agent_hp(self, agent_id, hp):
        self.agents[agent_id].health = hp

    def reset(self, options: Optional[Dict[str, Any]] = None):
        """Reset the game engine."""
        options = options or {}
        self.round_id = options.get("round_id", uuid.uuid4()) 
        self.logger = GameStateLogger(self.round_id, self.expected_agent_ids)
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
        self._terminal_reward_granted = False
        self.game_time_limit = GAME_TIME_LIMIT
        self._bomb_field = None
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
