from panda3d.core import Vec3, LineSegs
from panda3d.bullet import BulletCapsuleShape, ZUp, BulletCharacterControllerNode
import numpy as np
from .utils import Region, Team, vec_distance, Direction, AgentStats, get_opposite_team, bomb_status_to_onehot, BombStatus, Weapon, tramsform_to_minimap
from .config import *
from typing import Optional, List
import math
from direct.gui.OnscreenText import OnscreenText


class Agent:
    def __init__(self, engine, agent_id:str, 
                 team, spawn_mode:str, init_pos:Optional[Vec3], init_waypoint_id:int,
                 weapon: Optional[Weapon] = None,
                 has_armor: bool = False,
                 has_helmet: bool = False):
        # Core attributes
        self.engine = engine
        self.agent_id: str = agent_id
        self.team = team
        self.spawn_mode: str = spawn_mode
        self.init_pos: Optional[Vec3] = init_pos
        self.init_waypoint_id: int = init_waypoint_id
        self.weapon = weapon if weapon is not None else Weapon.AK_47 if team == Team.T else Weapon.M4A1
        self.has_armor = has_armor
        self.has_helmet = has_helmet

        # Game state
        self.is_alive: bool = True
        self.health: int = AGENT_MAX_HEALTH
        self.grenade = 1
        self.stats: AgentStats = AgentStats()
        self.waypoint_path: List[int] = []
        self.view_angle: Optional[Vec3] = 90

        # Movement State
        self.movement_speed: float = AGENT_MOVEMENT_SPEED
        self.current_waypoint_id: int = None
        self.target_pos: Optional[Vec3] = None
        self.target_waypoint: Optional[dict] = None
        self.current_direction: Optional[Direction] = None
        self.prev_position: Optional[Vec3] = None

        # Action State
        self.is_stopping: bool = False
        self.stopping_tick: Optional[int] = 0
        self.prev_decision_tick: Optional[int] = None
        self.current_reward: float = 0
        self.death_position: Optional[np.ndarray] = None

        self._setup_agent_model()
        self._setup_id_text()
        
        if engine.debug_mode:
            self._create_debug_visualization()

    @property
    def is_t(self):
        return self.team == Team.T

    @property
    def is_ct(self):
        return self.team == Team.CT
    
    @property
    def is_at_bomb_site(self):
        return self.region in [Region.A_BOMBSITE, Region.B_BOMBSITE]
    
    @property
    def is_near_bomb(self):
        distance_to_bomb = vec_distance(self.position, self.engine.current_bomb_position)
        return distance_to_bomb < BOMB_NEAR_DISTANCE_THRESHOLD

    @property
    def has_bomb(self):
        return self.engine.bomb_carrier == self.agent_id
    
    @property
    def region(self):
        return self.engine.waypoints.get_region_of_waypoint(self.current_waypoint['id'])

    @property
    def position(self):
        """Get the current position of the agent."""
        if not self.is_alive:
            return self.death_position
        hitbox_pos_center = self.node_path.getPos()
        hitbox_pos_center.z -= AGENT_HITBOX_HEIGHT/2
        return hitbox_pos_center
    
    @property
    def display_health(self):
        return max(0, self.health)
    
    @property
    def observation(self):
        pos = self.position
        health = self.display_health
        bomb_location = self.engine.current_bomb_position
        bomb_status_onehot = bomb_status_to_onehot(self.engine.bomb_status)
        teammate_healths = []
        teammate_positions = []
        for agent in self.engine.agents_by_team[self.team]:
            if agent.agent_id != self.agent_id:
                teammate_positions.extend(agent.position)
                teammate_healths.append(agent.display_health)
        return np.array([*pos, 
                         health, 
                         *bomb_location,
                         *bomb_status_onehot,
                         *teammate_positions,
                         *teammate_healths
                         ], dtype=np.float32)

    @property
    def action_mask(self):
        action_mask = [0] * 8
        for direction in Direction:
            if self.engine.waypoints.is_neighbor_valid(self.current_waypoint["id"], direction):
                action_mask[direction.value] = 1
        action_mask += [1] # stoping is always valid
        return np.array(action_mask, dtype=bool)

    @property
    def reward(self):
        return self.current_reward
    
    @property
    def termination(self):
        return not self.is_alive or self.engine.game_ended
    
    def reset(self):
        self.reset_agent_position()
        self.is_alive = True
        self.health = AGENT_MAX_HEALTH
        self.target_pos = None
        self.target_waypoint = None
        self.current_waypoint_id = None
        self.is_stopping = False
        self.stopping_tick = 0
        self.grenade = 1
        self.movement_speed = AGENT_MOVEMENT_SPEED
        self.current_reward = 0
        self.prev_position = None
        self.prev_decision_tick = None
        self.current_direction = None
        self.death_position = None
        self.waypoint_path = []
        self.stats.reset()

    def handle_automatic_game_actions(self):
        """Handle automatic game actions."""
        pass
        # if ENABLE_AGENT_SHOOTING:  # Only process shooting if enabled
        #     for enemy_agent in self.engine.agents_by_team[get_opposite_team(self.team)]:
        #         if not enemy_agent.is_alive:
        #             continue 
        #         if self.has_line_of_sight_to_agent(enemy_agent):
        #             successful_hit = random.random() > 0.7
        #             if successful_hit:
        #                 damage = self.engine.estimate_hit_damage(self, enemy_agent)
        #                 enemy_agent.health -= damage
        #                 self._draw_shooting_line(enemy_agent)
        #             break

        # if self.grenade > 0:
        #     self.engine.handle_grenade(self.agent_id, self.grenade)

    def handle_death(self):
        if self.health <= 0:
            self.die()
            self.engine.termination_queue.append(self.agent_id)

    def handle_bomb_actions(self):
        if not ENABLE_BOMB_ACTIONS:
            return
        
        if self.is_t and self.has_bomb and self.is_at_bomb_site:
            plant_waypoint_id = self.current_waypoint['id']
            self.engine.plant_bomb(plant_waypoint_id)
        elif self.is_ct and self.is_near_bomb and self.engine.bomb_has_planted:
            self.engine.defuse_bomb()

    def die(self):
        """When an agent dies, it's class stays to keep the stats going but its models are removed to protect other game logic"""
        self.death_position = self.position
        self.is_alive = False
        self.node_path.node().setLinearMovement(Vec3(0, 0, 0), True)
        self.remove_model_assets()

        if self.engine.bomb_carrier == self.agent_id:
            self.engine.bomb_carrier = None
            self.engine.bomb_status = BombStatus.Dropped
            self.engine.bomb_world_position = self.current_waypoint['pos']

    def reset_agent_position(self):
        if self.spawn_mode == "random":
            reset_waypoint = self.engine.waypoints.get_random_waypoint()
        elif self.spawn_mode == "random_spawn":
            if self.is_t:
                reset_waypoint = self.engine.waypoints.get_random_waypoint_in_region(Region.T_SPAWN)
            elif self.is_ct:
                reset_waypoint = self.engine.waypoints.get_random_waypoint_in_region(Region.CT_SPAWN)
        elif self.spawn_mode == "fixed":
            if self.init_pos is not None:
                reset_pos = self.init_pos
            else:
                reset_waypoint = self.engine.waypoints.get_waypoint_by_id(self.init_waypoint_id)
        else:
            raise ValueError(f"Invalid spawn mode: {self.spawn_mode}")
        reset_pos = reset_waypoint['pos'] if self.init_pos is None else self.init_pos
        reset_pos = reset_pos + Vec3(0, 0, INITIAL_HEIGHT_OFFSET)
        self.node_path.setPos(reset_pos)
        self.current_waypoint = reset_waypoint

    def set_move_target(self, direction: Direction, waypoint):
        """Set the target position based on the given action."""
        self.target_waypoint = waypoint
        self.target_pos = waypoint['pos']
        self.current_direction = direction
        self.waypoint_path.append(self.target_waypoint['id'])


    def set_stop_action(self):
        self.is_stopping = True
        self.target_waypoint = self.current_waypoint
        self.target_pos = self.current_waypoint['pos']
        self.waypoint_path.append(self.target_waypoint['id'])

    def update_movement(self):
        """Execute movement towards target using character controller."""
        if self._handle_stuck_agent():
            return
        
        if self._handle_stopping():
            return
        
        self._update_distance_stats()
        self._handle_movement()


    def _handle_stuck_agent(self):
        agent_is_stuck = self.prev_decision_tick is not None and \
            self.engine.physics_ticks - self.prev_decision_tick > STUCK_AGENT_TICK_THRESHOLD
        if agent_is_stuck:
            self.node_path.setPos(self.target_pos + Vec3(0, 0, INITIAL_HEIGHT_OFFSET))
            self.stats.stuck_count += 1
            return True
        return False
    
    def _handle_stopping(self):
        if self.is_stopping:
            self.node_path.node().setLinearMovement(Vec3(0, 0, 0), True)
            self.stopping_tick += 1
            if self.stopping_tick >= STOP_ACTION_TICK:
                self.stopping_tick = 0
                self.is_stopping = False
            return True
        return False
    
    def _update_distance_stats(self):
        current_pos = self.position
        if self.prev_position is not None:
            self.stats.total_xy_distance += vec_distance(self.prev_position.xy, current_pos.xy)
            self.stats.total_distance += vec_distance(self.prev_position, current_pos)
        self.prev_position = current_pos

    def _handle_movement(self):
        current_pos = self.position
        direction = self.target_pos - current_pos
        horizontal_direction = Vec3(direction.x, direction.y, 0)
        horizontal_direction.normalize()
        movement = horizontal_direction * self.movement_speed

        self._handle_jumping(current_pos)
        self.node_path.node().setLinearMovement(movement, True)

    def _handle_jumping(self, current_pos):
        height_difference = self.target_pos.z - current_pos.z
        if height_difference > AGENT_JUMP_HEIGHT_THRESHOLD and self.node_path.node().isOnGround():
            self.node_path.node().setMaxJumpHeight(AGENT_MAX_JUMP_HEIGHT)
            self.node_path.node().setJumpSpeed(AGENT_JUMP_SPEED)
            self.node_path.node().doJump()
            self.stats.total_jumps += 1

    def has_line_of_sight_to_agent(self, target_agent) -> bool:
        """Check if there is a clear line of sight to the target agent."""
        my_pos = self.position + Vec3(0, 0, AGENT_HITBOX_HEIGHT/2)
        target_pos = target_agent.position + Vec3(0, 0, AGENT_HITBOX_HEIGHT/2)
        result = self.engine.world.rayTestClosest(my_pos, target_pos)
        return result.hasHit() and result.getNode() == target_agent.node_path.node()

    def has_decision_request(self):
        """Check if the agent has requested a decision."""
        has_reached = self.has_reached_target()
        if has_reached and not self.is_stopping:
            # self.node_path.node().setLinearMovement(Vec3(0, 0, 0), True)
            # self.node_path.setPos(self.target_pos + Vec3(0, 0, INITIAL_HEIGHT_OFFSET))
            self.current_waypoint = self.target_waypoint
            if self.prev_decision_tick is not None:
                delta_tick = self.engine.physics_ticks - self.prev_decision_tick
                self.stats.decision_ticks.append((self.current_direction, delta_tick))
            self.prev_decision_tick = self.engine.physics_ticks
        return has_reached
    
    def has_reached_target(self):
        """Check if the agent has reached the target position."""
        xy_distance = vec_distance(self.position.xy, self.target_pos.xy)
        has_reached_xy = self.target_pos is not None and \
            xy_distance < TARGET_REACH_DISTANCE_THRESHOLD
        # has_reached_z = self.target_pos is not None and \
        #     abs(self.position.z - self.target_pos.z) < TARGET_REACH_DISTANCE_THRESHOLD_HEIGHT
        # return has_reached_xy and has_reached_z
        return has_reached_xy
    
    def is_colliding_map(self) -> bool:
        """Check if the agent is colliding with any environmental objects."""
        result = self.engine.world.contactTest(self.node_path.node())
        
        # Check each contact
        for contact in result.getContacts():
            # Get the two nodes involved in the contact
            node0 = contact.getNode0()
            node1 = contact.getNode1()
            
            # Check if either node is the environment (has COLLISION_BITMASK_ENVIRONMENT)
            if not (node0.getNetCollideMask() & COLLISION_BITMASK_ENVIRONMENT).isZero() or \
               not (node1.getNetCollideMask() & COLLISION_BITMASK_ENVIRONMENT).isZero():
                return True
                
        return False
    
    def _setup_agent_model(self):
        """Setup the agent's hitbox and character controller."""
        self.model = self.engine.loader.loadModel(AGENT_MODEL_PATH, noCache=NO_MODEL_CACHE)
        self.model.setScale(AGENT_SCALE)
        if self.is_t:
            self.model.setColorScale(*TERRORIST_COLOR)
        elif self.is_ct:
            self.model.setColorScale(*COUNTER_TERRORIST_COLOR)

        radius = AGENT_HITBOX_RADIUS
        height = AGENT_HITBOX_HEIGHT
        shape = BulletCapsuleShape(radius, height - 2*radius, ZUp)
        shape.setMargin(AGENT_HITBOX_MARGIN)
        node = BulletCharacterControllerNode(shape, radius, f'Agent_{self.agent_id}')
        node.setMaxSlope(math.radians(AGENT_MAX_SLOPE_DEG))
        self.node_path = self.engine.render.attachNewNode(node)
        self.model.reparentTo(self.node_path)
        self.model.setPos(0, 0, -AGENT_HITBOX_HEIGHT/2)
        self.node_path.setCollideMask(COLLISION_BITMASK_AGENTS)
        
        self.engine.world.attachCharacter(node)

    def _setup_id_text(self):
        """Setup the floating ID text above the agent."""
        text_node = self.engine.render.attachNewNode('text_node')
        text_node.setBillboardPointEye()
        text_node.reparentTo(self.node_path)
        text_node.setPos(0, 0, AGENT_HITBOX_HEIGHT*3/5)
        self.id_text = OnscreenText(
            text=str(self.agent_id),
            fg=(1, 1, 1, 1),
            scale=0.1,
            parent=text_node,
        )

    def _create_debug_visualization(self):
        # Create line segments for hitbox visualization
        ls = LineSegs()
        ls.setThickness(DEBUG_LINE_THICKNESS)
        ls.setColor(*DEBUG_HITBOX_COLOR)
        
        # Draw vertical line for height
        ls.moveTo(0, 0, -AGENT_HITBOX_HEIGHT/2)
        ls.drawTo(0, 0, AGENT_HITBOX_HEIGHT/2)
        
        # Draw circles at top and bottom
        segments = 16  # Number of segments to approximate circles
        for z in [-AGENT_HITBOX_HEIGHT/2, AGENT_HITBOX_HEIGHT/2]:  # Draw at both top and bottom
            for i in range(segments + 1):
                angle = (i / segments) * 2 * 3.14159
                x = AGENT_HITBOX_RADIUS * math.cos(angle)
                y = AGENT_HITBOX_RADIUS * math.sin(angle)
                if i == 0:
                    ls.moveTo(x, y, z)
                else:
                    ls.drawTo(x, y, z)

        debug_node = ls.create()
        debug_lines = self.node_path.attachNewNode(debug_node)
        
        self.engine.debug_manager.debug_lines[self.agent_id] = debug_lines

    def draw_shooting_line(self, target_agent):
        if self.engine.debug_mode:
            my_pos = self.position + Vec3(0, 0, AGENT_HITBOX_HEIGHT/2)
            target_pos = target_agent.position + Vec3(0, 0, AGENT_HITBOX_HEIGHT/2)
            # Draw a temporary line for one frame
            line_node = self.engine.debug_manager.draw_debug_line(
                my_pos, 
                target_pos, 
                color=(1, 1, 0, 1),
                thickness=3.0
            )
            self.engine.debug_manager.shooting_lines.append(line_node)

    def get_debug_info(self) -> str:
        """Format debug information for this agent."""
        pos = self.position
        # Set color based on agent state
        if not self.is_alive:
            color_tag = '\1red\1'  # Red for dead agents
        else:
            color_tag = '\1white\1'  # White for living agents
        debug_str = f"{color_tag}Agent {self.agent_id}: HP: {self.display_health} Armor: {self.has_armor} Helmet: {self.has_helmet} Weapon: {self.weapon.name}\n"
        debug_str += f"  Position: ({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f}) ID: {self.current_waypoint['id']}\n"
        if self.target_pos:
            debug_str += f"  Target: ({self.target_pos.x:.2f}, {self.target_pos.y:.2f}, {self.target_pos.z:.2f}) ID: {self.target_waypoint['id']}\n"
        debug_str += "\n"
        return debug_str

    def remove_model_assets(self):
        """Clean up the agent's resources and remove it from the game world."""
        self.engine.world.removeCharacter(self.node_path.node())

        if self.engine.debug_mode and self.agent_id in self.engine.debug_manager.debug_lines:
            self.engine.debug_manager.debug_lines[self.agent_id].removeNode()
            del self.engine.debug_manager.debug_lines[self.agent_id]

        self.model.removeNode()
        self.node_path.removeNode()
        self.id_text.destroy()

    