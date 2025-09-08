from panda3d.core import Vec3, LineSegs, TextNode, TextProperties, TextPropertiesManager, CardMaker, TransparencyAttrib
from direct.gui.OnscreenText import OnscreenText
from .config import *
from .utils import tramsform_to_minimap

class DebugManager:
    def __init__(self, engine):
        self.engine = engine
        self.debug_lines = {}
        
        # Individual toggle states for debug features
        # Initialize based on engine flags
        self.show_debug_text = engine.debug_mode
        self.show_minimap = engine.show_minimap
        self.show_waypoints = engine.show_waypoints
        self.show_agent_paths = engine.debug_mode  # Agent paths are part of debug mode
        
        # Only initialize components that are enabled
        if self.show_debug_text:
            self._init_debug_display()
        else:
            self.debug_text = None
            self.system_text = None
            
        self.waypoints_root = None
        self.shooting_lines = []
        self.minimap_dots = []
        
        if self.show_minimap:
            self._init_minimap()
        else:
            self.minimap_np = None

    def reset(self):
        for line in self.shooting_lines:
            line.removeNode()
        self.shooting_lines = []
        for dot in self.minimap_dots:
            dot.removeNode()
        self.minimap_dots = []

    def toggle_debug_text(self):
        """Toggle debug text display on/off."""
        if not self.engine.render_mode:
            return
        self.show_debug_text = not self.show_debug_text
        if not self.show_debug_text and self.debug_text:
            self.debug_text.setText("")
            self.system_text.setText("")
        elif self.show_debug_text and not self.debug_text:
            self._init_debug_display()
        print(f"Debug text: {'ON' if self.show_debug_text else 'OFF'}")

    def toggle_minimap(self):
        """Toggle minimap display on/off."""
        if not self.engine.render_mode:
            return
        self.show_minimap = not self.show_minimap
        if not self.show_minimap and self.minimap_np:
            self.minimap_np.hide()
            for dot in self.minimap_dots:
                dot.removeNode()
            self.minimap_dots = []
        elif self.show_minimap and not self.minimap_np:
            self._init_minimap()
        elif self.show_minimap and self.minimap_np:
            self.minimap_np.show()
        print(f"Minimap: {'ON' if self.show_minimap else 'OFF'}")

    def toggle_waypoints(self):
        """Toggle waypoint visualization on/off."""
        if not self.engine.render_mode:
            return
        self.show_waypoints = not self.show_waypoints
        if not self.show_waypoints and self.waypoints_root:
            self.waypoints_root.hide()
        elif self.show_waypoints and not self.waypoints_root:
            self.create_waypoint_visualization()
        elif self.show_waypoints and self.waypoints_root:
            self.waypoints_root.show()
        print(f"Waypoints: {'ON' if self.show_waypoints else 'OFF'}")

    def toggle_agent_paths(self):
        """Toggle agent path visualization on/off."""
        if not self.engine.render_mode:
            return
        self.show_agent_paths = not self.show_agent_paths
        if not self.show_agent_paths:
            for agent_id, line_node in self.debug_lines.items():
                line_node.hide()
        else:
            for agent_id, line_node in self.debug_lines.items():
                line_node.show()
        print(f"Agent paths: {'ON' if self.show_agent_paths else 'OFF'}")

    def toggle_all_debug(self):
        """Toggle all debug features on/off."""
        if not self.engine.render_mode:
            return
        # If any are on, turn all off; if all are off, turn all on
        all_off = not (self.show_debug_text or self.show_minimap or self.show_waypoints or self.show_agent_paths)
        
        target_state = all_off
        if target_state != self.show_debug_text:
            self.toggle_debug_text()
        if target_state != self.show_minimap:
            self.toggle_minimap()
        if target_state != self.show_waypoints:
            self.toggle_waypoints()
        if target_state != self.show_agent_paths:
            self.toggle_agent_paths()
        
        print(f"All debug features: {'ON' if target_state else 'OFF'}")

    def _init_debug_display(self):
        """Initialize debug text display."""
        tp_mgr = TextPropertiesManager.getGlobalPtr()
        tp_red = TextProperties()
        tp_red.setTextColor(1, 0, 0, 1)  # Red
        tp_mgr.setProperties("red", tp_red)  # Register as "red"

        tp_white = TextProperties()
        tp_white.setTextColor(1, 1, 1, 1)  # White
        tp_mgr.setProperties("white", tp_white)  # Register as "white"

        # Calculate positions based on aspect ratio
        aspect_ratio = self.engine.getAspectRatio()
        left_edge_x = -aspect_ratio + 0.05  # Small margin from left edge
        right_edge_x = aspect_ratio - 0.05  # Small margin from right edge

        self.debug_text = OnscreenText(
            text="",
            style=1,
            fg=(1, 1, 1, 1),  # White color
            pos=(left_edge_x, 0.9),   # Top left position (dynamic)
            align=TextNode.ALeft,
            scale=0.04
        )
        
        self.system_text = OnscreenText(
            text="",
            style=1,
            fg=(1, 1, 1, 1),  # White color
            pos=(right_edge_x, -0.55),    # Bottom right position (dynamic)
            align=TextNode.ARight,
            scale=0.04
        )
        
        self.engine.taskMgr.add(self.update_debug_text, "UpdateDebugText")
        self.engine.taskMgr.add(self.update_minimap, "UpdateMinimap")

    def _init_minimap(self):
        """Initialize the minimap."""
        cm = CardMaker("minimap")
        cm.setFrame(-1, 1, -1, 1)  # Full quad (-1 to 1)

        self.minimap_np = self.engine.aspect2d.attachNewNode(cm.generate())
        self.minimap_np.setTransparency(TransparencyAttrib.M_alpha)
        self.minimap_np.setScale(MINIMAP_RATIO)

        aspect_ratio = self.engine.getAspectRatio()  # Gets the window's width-to-height ratio

        right_edge_x = aspect_ratio  # The rightmost screen position in aspect2d
        top_edge_y = 1  # Topmost position in aspect2d (always 1)

        self.minimap_np.setPos(right_edge_x - MINIMAP_RATIO, 0, top_edge_y - MINIMAP_RATIO)
        minimap_texture = self.engine.loader.loadTexture(MINIMAP_IMAGE_PATH)
        self.minimap_image_size = minimap_texture.getXSize()
        self.minimap_np.setTexture(minimap_texture)

    def update_minimap(self, task):
        """Updates player dots on the minimap"""
        if not self.show_minimap or self.engine.render_mode != "spectator" or not self.minimap_np:
            return task.cont
        
        if not SHOW_TRACE:
            for dot in self.minimap_dots:
                dot.removeNode()
            self.minimap_dots = []
        
        for agent_id, agent in self.engine.agents.items():
            if not agent.is_alive:
                color = DEAD_AGENT_COLOR
            else:
                color = TERRORIST_COLOR if agent.is_t else COUNTER_TERRORIST_COLOR
            self.add_minimap_dot(agent.position, color)

        self.add_minimap_dot(self.engine.current_bomb_position, BOMB_COLOR)
        return task.cont
    
    def add_minimap_dot(self, world_pos, color):
        """Adds a dot for a player based on transformed coordinates"""
        x, y = tramsform_to_minimap(world_pos, self.minimap_image_size)

        cm = CardMaker("dot")
        cm.setFrame(-MINIMAP_DOT_SIZE, MINIMAP_DOT_SIZE, -MINIMAP_DOT_SIZE, MINIMAP_DOT_SIZE)  # Small square dot

        dot_np = self.minimap_np.attachNewNode(cm.generate())
        dot_texture = self.engine.loader.loadTexture(MINIMAP_DOT_TEXTURE_PATH)
        dot_np.setTexture(dot_texture)
        dot_np.setTransparency(TransparencyAttrib.M_alpha)
        dot_np.setPos(x, 0, y)
        dot_np.setColor(*color)
        self.minimap_dots.append(dot_np)

    def update_debug_text(self, task):
        """Update the debug text display."""
        if not self.show_debug_text or self.engine.render_mode != "spectator" or not self.debug_text:
            return task.cont
        
        # System information (time and camera)
        system_str = self._get_time_debug_info()
        system_str += "\n" + self._get_camera_debug_info()
        self.system_text.setText(system_str)

        # Agent information (stays at top left)
        debug_str = self._get_agent_debug_info()
        self.debug_text.setText(debug_str)
        
        return task.cont

    def _get_time_debug_info(self):
        """Get formatted time debug information."""
        avg_agent_decisions = self.engine.total_agent_decision_requests / len(self.engine.agents) if len(self.engine.agents) > 0 else 0
        avg_ticks_per_decision = self.engine.physics_ticks / avg_agent_decisions if avg_agent_decisions > 0 else 0
        
        # time_str = f"FPS: {self.engine.clock.getAverageFrameRate():.1f}\n"
        # time_str += f"RealWorld Time: {self.engine.clock.getRealTime():.2f}s\n"
        time_str = f"Physics Time: {self.engine.game_time:.2f}s\n"
        time_str += f"Total Ticks: {self.engine.physics_ticks}\n"
        time_str += f"Total Agent Decisions: {self.engine.total_agent_decision_requests}\n"
        time_str += f"Avg Agent Decisions: {avg_agent_decisions:.2f}\n"
        time_str += f"Avg Ticks Per Decision: {avg_ticks_per_decision:.2f}\n"
        time_str += f"Time Scale Setting: {self.engine.time_scale}x\n\n"
        
        # Add debug controls help
        # time_str += "Debug Controls:\n"
        # time_str += "F1: Text | F2: Minimap | F3: Waypoints\n"
        # time_str += "F4: Agent Paths | F5: All Debug"
        return time_str

    def _get_camera_debug_info(self):
        """Get formatted camera debug information."""
        cam_pos = self.engine.cam.getPos()
        cam_hpr = self.engine.cam.getHpr()
        
        # Determine camera facing direction
        heading = cam_hpr.x % 360  # Normalize to 0-360 range
        pitch = cam_hpr.y % 360
        
        # Determine primary direction based on pitch
        if 45 <= pitch <= 135:
            axis = "+Z"
            cardinal = direction = "Up"
        elif 225 <= pitch <= 315:
            axis = "-Z"
            cardinal = direction = "Down"
        # If not primarily looking up/down, determine horizontal direction
        else:
            if 45 <= heading < 135:
                axis = "-X"
                cardinal = "West"
                direction = "Left"
            elif 135 <= heading < 225:
                axis = "-Y"
                cardinal = "South"
                direction = "Back"
            elif 225 <= heading < 315:
                axis = "+X"
                cardinal = "East"
                direction = "Right"
            else:  # 315-360 or 0-45
                axis = "+Y"
                cardinal = "North"
                direction = "Forward"

        debug_str = f"Camera Facing: {axis} ({direction}) ({cardinal})\n"
        debug_str += f"Camera Position: ({cam_pos.x:.2f}, {cam_pos.y:.2f}, {cam_pos.z:.2f})\n"
        debug_str += f"Camera Rotation: ({cam_hpr.x:.2f}, {cam_hpr.y:.2f}, {cam_hpr.z:.2f})"
        return debug_str

    def _get_agent_debug_info(self):
        """Get formatted agent debug information."""
        # Separate agents by team
        t_agents = [(id, agent) for id, agent in self.engine.agents.items() if agent.is_t]
        ct_agents = [(id, agent) for id, agent in self.engine.agents.items() if agent.is_ct]
        
        debug_str = self._format_team_debug_info("T Agents:", t_agents)
        debug_str += self._format_team_debug_info("CT Agents:", ct_agents)
        return debug_str

    def _format_team_debug_info(self, team_name, agents):
        debug_str = f"{team_name}\n"
        for agent_id, agent in agents[:5]:
            debug_str += agent.get_debug_info()
        
        if len(agents) > 5:
            debug_str += f"... and {len(agents) - 5} more {team_name}\n"
        
        return debug_str

    def update_agent_debug_line(self, agent_id):
        """Update the debug line for an agent."""
        agent = self.engine.agents[agent_id]
        current_pos = agent.position
        target_pos = agent.target_pos
        
        # Clear existing line
        if agent_id in self.debug_lines:
            self.debug_lines[agent_id].removeNode()
        
        # Create new line if there's a target and agent paths are enabled
        if target_pos is not None and self.show_agent_paths:
            ls = LineSegs()
            ls.setThickness(3.0)  # Use your DEBUG_AGENT_PATH_THICKNESS constant here
            ls.setColor(1, 0, 0, 1)  # Use your DEBUG_AGENT_PATH_COLOR constant here
            
            ls.moveTo(current_pos)
            ls.drawTo(target_pos)
            
            debug_node = ls.create()
            self.debug_lines[agent_id] = self.engine.render.attachNewNode(debug_node)
        else:
            # Create empty node if paths are disabled
            self.debug_lines[agent_id] = self.engine.render.attachNewNode("empty_debug_line")

    def create_waypoint_visualization(self) -> None:
        """Create visual debug markers for waypoints."""
        self.waypoints_root = self.engine.render.attachNewNode("waypoints")
        ls = LineSegs()
        
        # Draw waypoint markers
        self._draw_waypoint_markers(ls)
        # Draw connection lines
        self._draw_waypoint_connections(ls)
        
        waypoint_node = ls.create()
        self.waypoints_root.attachNewNode(waypoint_node)
        
        # Set initial visibility based on toggle state
        if not self.show_waypoints:
            self.waypoints_root.hide()

    def _draw_waypoint_markers(self, ls: LineSegs) -> None:
        """Draw debug markers for each waypoint."""
        for node_id, attr in self.engine.waypoints.graph.nodes(data=True):
            ls.setThickness(DEBUG_LINE_THICKNESS)
            ls.setColor(*DEBUG_WAYPOINT_COLOR)
            pos = attr['pos']
            x, y, z = pos.x, pos.y, pos.z
            
            size = DEBUG_WAYPOINT_SIZE 
            ls.moveTo(x, y, z - size)
            ls.drawTo(x, y, z + size)
            ls.moveTo(x - size, y, z)
            ls.drawTo(x + size, y, z)
            ls.moveTo(x, y - size, z)
            ls.drawTo(x, y + size, z)

    def _draw_waypoint_connections(self, ls: LineSegs) -> None:
        """Draw connection lines between waypoints."""
        for node_id, attr in self.engine.waypoints.graph.nodes(data=True):
            pos = attr['pos']
            x, y, z = pos.x, pos.y, pos.z
            
            for neighbor_node in self.engine.waypoints.get_neighbors(node_id):
                neighbor_pos = neighbor_node['pos']
                if not self.engine.waypoints.graph.has_edge(neighbor_node['id'], node_id):
                    # Unidirectional edge - draw red line with arrow
                    ls.setThickness(DEBUG_UNIDIRECTIONAL_EDGE_THICKNESS)
                    ls.setColor(*DEBUG_UNIDIRECTIONAL_EDGE_COLOR)
                    ls.moveTo(x, y, z)
                    ls.drawTo(neighbor_pos.x, neighbor_pos.y, neighbor_pos.z)
                    
                    # Draw arrow head
                    direction = (neighbor_pos - pos).normalized()
                    arrow_size = 0.2  # Size of arrow head
                    arrow_pos = Vec3(
                        neighbor_pos.x - direction.x * arrow_size,
                        neighbor_pos.y - direction.y * arrow_size,
                        neighbor_pos.z - direction.z * arrow_size
                    )
                    
                    # Calculate perpendicular vectors for arrow head
                    up = Vec3(0, 0, 1)
                    right = direction.cross(up).normalized()
                    if right.length() < 0.1:  # If direction is vertical, use different up vector
                        up = Vec3(0, 1, 0)
                        right = direction.cross(up).normalized()
                    up = right.cross(direction).normalized()
                    
                    # Draw arrow head
                    ls.moveTo(neighbor_pos.x, neighbor_pos.y, neighbor_pos.z)
                    ls.drawTo(arrow_pos.x + right.x * arrow_size, 
                            arrow_pos.y + right.y * arrow_size,
                            arrow_pos.z + right.z * arrow_size)
                    
                    ls.moveTo(neighbor_pos.x, neighbor_pos.y, neighbor_pos.z)
                    ls.drawTo(arrow_pos.x - right.x * arrow_size,
                            arrow_pos.y - right.y * arrow_size,
                            arrow_pos.z - right.z * arrow_size)
                else:
                    ls.setThickness(DEBUG_BIDIRECTIONAL_EDGE_THICKNESS)
                    ls.setColor(*DEBUG_BIDIRECTIONAL_EDGE_COLOR)
                    ls.moveTo(x, y, z)
                    ls.drawTo(neighbor_pos.x, neighbor_pos.y, neighbor_pos.z)

    def draw_debug_line(self, start_pos, end_pos, color=(1, 0, 0, 1), thickness=3.0):
        """Draw a debug line between two points.
        
        Args:
            start_pos (Vec3 or tuple): Starting position (x, y, z)
            end_pos (Vec3 or tuple): Ending position (x, y, z)
            color (tuple): RGBA color tuple (default: red)
            thickness (float): Line thickness (default: 3.0)
            
        Returns:
            NodePath: The created line's NodePath
        """
        ls = LineSegs()
        ls.setThickness(thickness)
        ls.setColor(*color)
        
        # Convert tuples to Vec3 if necessary
        start = Vec3(*start_pos) if not isinstance(start_pos, Vec3) else start_pos
        end = Vec3(*end_pos) if not isinstance(end_pos, Vec3) else end_pos
        
        ls.moveTo(start)
        ls.drawTo(end)
        
        debug_node = ls.create()
        return self.engine.render.attachNewNode(debug_node)

    def draw_connected_debug_lines(self, points, color=(1, 0, 0, 1), thickness=3.0):
        """Draw connected debug lines through a sequence of points.
        
        Args:
            points (list): List of Vec3 or (x, y, z) tuples
            color (tuple): RGBA color tuple (default: red)
            thickness (float): Line thickness (default: 3.0)
            
        Returns:
            NodePath: The created lines' NodePath
        """
        if len(points) < 2:
            return None
            
        ls = LineSegs()
        ls.setThickness(thickness)
        ls.setColor(*color)
        
        # Convert first point and move to it
        start = Vec3(*points[0]) if not isinstance(points[0], Vec3) else points[0]
        ls.moveTo(start)
        
        # Draw lines to subsequent points
        for point in points[1:]:
            pos = Vec3(*point) if not isinstance(point, Vec3) else point
            ls.drawTo(pos)
        
        debug_node = ls.create()
        return self.engine.render.attachNewNode(debug_node)