from panda3d.core import WindowProperties, Vec3
from .config import *
import sys


class SpectatorCamera:
    def __init__(self, engine):
        self.engine = engine
        self.base_camera_speed = SPECTATOR_BASE_SPEED
        self.sprint_multiplier = SPECTATOR_SPRINT_MULTIPLIER
        self.sensitivity = SPECTATOR_MOUSE_SENSITIVITY
        self.key_map = {
            "w": False, "s": False, "a": False, 
            "d": False, "shift": False
        }

        self._setup_controls()
        self._setup_tasks()

    def _setup_controls(self):
        props = WindowProperties()
        props.setCursorHidden(WINDOW_PROPERTIES['cursor_hidden'])
        props.setMouseMode(WindowProperties.M_relative if WINDOW_PROPERTIES['mouse_mode'] == 'relative' else WindowProperties.M_absolute)
        self.engine.win.requestProperties(props)

        for key in self.key_map:
            self.engine.accept(key, self.set_key, [key, True])
            self.engine.accept(f"{key}-up", self.set_key, [key, False])

        self.engine.accept("mouse1", self.enable_mouse_look)
        self.engine.accept('escape', self.cleanup_and_exit)
        # Add new control bindings
        self.engine.accept('space', self.toggle_simulation_pause)
        self.engine.accept('arrow_right', self.request_single_step)

        # Add time scale control bindings
        self.engine.accept('arrow_up', self.adjust_time_scale, [2.0])  # Double the time scale
        self.engine.accept('arrow_down', self.adjust_time_scale, [0.5])  # Halve the time scale

        # Debug visualization toggle bindings (only work when debug_mode is enabled)
        self.engine.accept('f1', self.toggle_debug_text)  # Toggle debug text
        self.engine.accept('f2', self.toggle_minimap)     # Toggle minimap
        self.engine.accept('f3', self.toggle_waypoints)   # Toggle waypoints
        self.engine.accept('f4', self.toggle_agent_paths) # Toggle agent paths
        self.engine.accept('f5', self.toggle_all_debug)   # Toggle all debug features

    def _setup_tasks(self):
        self.engine.taskMgr.add(self.update_camera_movement, "UpdateCameraMovement")
        self.engine.taskMgr.add(self.update_mouse_look, "UpdateMouseLook")

    def set_key(self, key, value):
        """Update key state for movement."""
        self.key_map[key] = value

    def update_camera_movement(self, task):
        """Update camera position based on keyboard input."""
        dt = self.engine.clock.getDt()
        
        # Get the camera's orientation vectors
        forward = self.engine.cam.getQuat().getForward()
        right = self.engine.cam.getQuat().getRight()
        
        # Calculate movement direction
        direction = Vec3(0, 0, 0)
        if self.key_map["w"]: direction += forward
        if self.key_map["s"]: direction -= forward
        if self.key_map["a"]: direction -= right
        if self.key_map["d"]: direction += right

        # Move if there's any input
        if direction.length() > 0:
            direction.normalize()
            speed = self.base_camera_speed * (self.sprint_multiplier if self.key_map["shift"] else 1.0)
            movement = direction * speed * dt
            self.engine.cam.setPos(self.engine.cam.getPos() + movement)

        return task.cont

    def update_mouse_look(self, task):
        """Update camera rotation based on mouse movement."""
        if not self.engine.mouseWatcherNode.hasMouse():
            return task.cont

        md = self.engine.win.getPointer(0)
        dx = md.getX() - self.engine.win.getXSize() / 2
        dy = md.getY() - self.engine.win.getYSize() / 2

        h = self.engine.cam.getH() - dx * self.sensitivity
        p = self.engine.cam.getP() - dy * self.sensitivity
        self.engine.cam.setHpr(h, p, 0)

        self.engine.win.movePointer(0, self.engine.win.getXSize() // 2, self.engine.win.getYSize() // 2)
        return task.cont

    def enable_mouse_look(self):
        """Enable mouse look controls."""
        props = WindowProperties()
        props.setCursorHidden(True)
        props.setMouseMode(WindowProperties.M_relative)
        self.engine.win.requestProperties(props)

    def cleanup_and_exit(self):
        """Clean up and exit the game when ESC is pressed."""
        print("Exiting game")
        sys.exit()

    def adjust_time_scale(self, multiplier):
        self.engine.time_scale *= multiplier

    def toggle_simulation_pause(self):
        """Toggle simulation pause state."""
        self.engine.simulation_paused = not self.engine.simulation_paused

    def request_single_step(self):
        """Request a single physics step when paused."""
        if self.engine.simulation_paused:
            self.engine.single_step_requested = True

    def toggle_debug_text(self):
        """Toggle debug text display."""
        if self.engine.debug_manager:
            self.engine.debug_manager.toggle_debug_text()

    def toggle_minimap(self):
        """Toggle minimap display."""
        if self.engine.debug_manager:
            self.engine.debug_manager.toggle_minimap()

    def toggle_waypoints(self):
        """Toggle waypoint visualization."""
        if self.engine.debug_manager:
            self.engine.debug_manager.toggle_waypoints()

    def toggle_agent_paths(self):
        """Toggle agent path visualization."""
        if self.engine.debug_manager:
            self.engine.debug_manager.toggle_agent_paths()

    def toggle_all_debug(self):
        """Toggle all debug features."""
        if self.engine.debug_manager:
            self.engine.debug_manager.toggle_all_debug()

