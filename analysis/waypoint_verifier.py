from game_engine import CSGOEngine
from utils import Direction
import networkx as nx
from typing import Set, Tuple
from panda3d.core import LineSegs
from tqdm import tqdm
from config import PHYSICS_STEP

class WaypointVerifier:
    def __init__(self, render_mode=None, waypoint_data_path=None):
        """Initialize the verifier with a game engine instance."""
        self.engine = CSGOEngine(num_team_agents=1, render_mode=render_mode, debug_mode=True, waypoint_data_path=waypoint_data_path)
        self.invalid_connections: Set[Tuple[int, int]] = set()
        self.invalid_nodes: Set[int] = set()
        self.verification_timeout = 5.0  # seconds
        self.physics_step = PHYSICS_STEP
        self.max_steps = int(self.verification_timeout / self.physics_step)

    def update_agent_simulation(self, agent_id: str, max_steps: int) -> bool:
        """
        Run physics simulation for a single agent until it reaches target or times out.
        Returns True if agent reached target, False otherwise.
        """
        agent = self.engine.agents[agent_id]
        
        for _ in range(max_steps):
            # Update agent movement
            agent.update_movement()
            
            # Run physics simulation
            self.engine.process_physics_step()
            
            # Update taskMgr if in render mode
            if self.engine.render_mode is not None and _ % 1 == 0:
                self.engine.handle_rendering()
            
            if agent.has_reached_target():
                return True
        return False

    def verify_connection(self, from_id: int, to_id: int) -> bool:
        """
        Verify if an agent can traverse from one waypoint to another and back.
        Returns True if connection is valid in both directions, False otherwise.
        """
        # First direction: from_id -> to_id
        if not self._verify_single_direction(from_id, to_id):
            return False
            
        # # Second direction: to_id -> from_id
        # if not self._verify_single_direction(to_id, from_id):
        #     return False

        # if not self._verify_single_direction_two_step_path(from_id, to_id):
        #     return False
            
        return True
        
    def _verify_single_direction(self, start_id: int, end_id: int) -> bool:
        """
        Helper method to verify connection in a single direction.
        Returns True if connection is valid, False otherwise.
        """
        # Remove any existing test agent
        agent_id = "T_0"
        if agent_id in self.engine.agents:
            self.engine.agents[agent_id].remove_model_assets()

        target_waypoint = self.engine.waypoints.get_waypoint_by_id(end_id)
        target_pos = target_waypoint['pos']
        
        
        self.engine.add_agent(
            agent_id=agent_id,
            spawn_mode="fixed",
            init_waypoint_id=start_id
        )
        self.engine.agents[agent_id].reset()

        # if self.engine.agents[agent_id].is_colliding_map():
        #     self.invalid_nodes.add(start_id)
        #     return False

        # Run physics simulation for 20 steps to settle the agent
        # for _ in range(30):
        #     self.engine.process_physics_step()
        #     if self.engine.render_mode is not None:
        #         self.engine.handle_rendering()


        self.engine.agents[agent_id].set_move_target(Direction.FORWARD, target_waypoint)
        
        # Run custom simulation loop
        is_valid = self.update_agent_simulation(agent_id, self.max_steps)
        
        # Clean up the test agent
        self.engine.agents[agent_id].remove_model_assets()
        self.engine.agents = {}
        return is_valid
    
    def _verify_single_direction_two_step_path(self, start_id: int, end_id: int) -> bool:
        """
        Helper method to verify connection in a single direction.
        Returns True if connection is valid, False otherwise.
        """
        # Remove any existing test agent
        agent_id = "T_0"
        if agent_id in self.engine.agents:
            self.engine.agents[agent_id].remove_model_assets()

        middle_waypoint = self.engine.waypoints.get_waypoint_by_id(start_id)
        target_waypoint = self.engine.waypoints.get_waypoint_by_id(end_id)
        predecessor_waypoints = list(self.engine.waypoints.graph.predecessors(start_id))

        for predecessor_waypoint in predecessor_waypoints:
            self.engine.add_agent(
                agent_id=agent_id,
                spawn_mode="fixed",
                init_waypoint_id=predecessor_waypoint
            )
            self.engine.agents[agent_id].reset()
            self.engine.agents[agent_id].set_move_target(Direction.FORWARD, middle_waypoint)
        
            is_middle_valid = self.update_agent_simulation(agent_id, self.max_steps)
            if is_middle_valid:
                self.engine.agents[agent_id].set_move_target(Direction.FORWARD, target_waypoint)
                is_end_valid = self.update_agent_simulation(agent_id, self.max_steps)
                if not is_end_valid:
                    return False
            
            self.engine.agents[agent_id].remove_model_assets()
            self.engine.agents = {}
        return True

    def verify_all_connections(self) -> None:
        """Verify all connections in the waypoint graph."""
        total_edges = len(self.engine.waypoints.graph.edges())
        
        # Initialize progress bar
        pbar = tqdm(total=total_edges, desc="Verifying connections")
        invalid_count = 0
        
        for from_id, to_id in self.engine.waypoints.graph.edges():
            # print(from_id, "->" ,to_id)
            # if from_id < 3122:
            #     continue
            if not self.verify_connection(from_id, to_id):
                self.invalid_connections.add((from_id, to_id))
                invalid_count += 1
            
            # Update progress bar with current invalid count
            pbar.set_postfix({'Invalid': invalid_count})
            pbar.update(1)
        
        pbar.close()

    def remove_invalid_connections(self) -> None:
        """Remove all invalid connections from the waypoint graph and ensure strong connectivity."""
        removed_edges = 0
        removed_nodes = 0

        # First remove invalid nodes
        for node_id in self.invalid_nodes:
            self.engine.waypoints.remove_waypoint(node_id)
            removed_nodes += 1

        # Then remove invalid edges
        for from_id, to_id in self.invalid_connections:
            if self.engine.waypoints.graph.has_edge(from_id, to_id):
                self.engine.waypoints.remove_connection(from_id, to_id)
                removed_edges += 1

        # Find the largest strongly connected component
        strong_components = list(nx.strongly_connected_components(self.engine.waypoints.graph))
        if not strong_components:
            print("Warning: No strongly connected components found!")
            return
        
        largest_component = max(strong_components, key=len)
        
        # Remove all nodes not in the largest strongly connected component
        nodes_to_remove = set(self.engine.waypoints.graph.nodes()) - largest_component
        for node in nodes_to_remove:
            removed_edges += self.engine.waypoints.graph.degree(node)  # Count edges that will be removed with this node
            self.engine.waypoints.remove_waypoint(node)
            removed_nodes += 1

        print(f"\nRemoval statistics:")
        print(f"Removed {removed_edges} edges in total")
        print(f"Removed {removed_nodes} nodes in total")
        print(f"Remaining graph has {len(self.engine.waypoints.graph.nodes)} nodes and {len(self.engine.waypoints.graph.edges)} edges")

    def visualize_waypoints(self, limit=100):
        """Create 3D scatter plots of waypoint positions from different angles."""
        for i, (from_id, to_id) in enumerate(self.engine.waypoints.graph.edges()):
            waypoint = self.engine.waypoints.get_waypoint_by_id(from_id)
            waypoint_pos = self.engine.waypoints.get_waypoint_by_id(from_id, return_pos=True)
            to_waypoint = self.engine.waypoints.get_waypoint_by_id(to_id)
            to_waypoint_pos = self.engine.waypoints.get_waypoint_by_id(to_id, return_pos=True)
            ls = LineSegs()
            ls.setThickness(3.0)
            ls.setColor(1, 0, 0, 1)
            ls.moveTo(waypoint_pos)
            ls.drawTo(to_waypoint_pos)
            debug_node = ls.create()
            self.engine.render.attachNewNode(debug_node)


def main():
    # Initialize verifier
    waypoint_data_path = "assets/WaypointDust2Verified_p3d_clean.json"
    verifier = WaypointVerifier(render_mode=None, waypoint_data_path=waypoint_data_path)
    # verifier = WaypointVerifier(render_mode="spectator", waypoint_data_path=waypoint_data_path)

    verifier.verify_all_connections()
    verifier.remove_invalid_connections()
    
    # Print graph statistics
    graph = verifier.engine.waypoints.graph
    print(f"\nGraph statistics after verification:")
    print(f"Number of nodes: {len(graph.nodes)}")
    print(f"Number of edges: {len(graph.edges)}")
    print(f"Graph is strongly connected: {nx.is_strongly_connected(graph)}")
    print(f"Graph is weakly connected: {nx.is_weakly_connected(graph)}")

    # Verify graph integrity
    verifier.engine.waypoints.verify_graph_integrity(verbose=True)

    # Save verified graph
    output_path = waypoint_data_path.replace('.json', '_verified.json')
    # output_path = waypoint_data_path.replace('.json', '_two_step_verified.json')
    verifier.engine.waypoints.export_to_json(output_path)
    print(f"\nVerified waypoint graph saved to: {output_path}")
    

if __name__ == "__main__":
    main()