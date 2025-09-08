import networkx as nx
from panda3d.core import Vec3
import json
from collections import defaultdict
import random
from .utils import Direction, vec_distance, Region
import plotly.graph_objects as go
from .config import WAYPOINT_DATA_PATH, STOP_ACTION_INDEX


ADDITIONAL_WAYPOINTS = [
    {
        "nodeid": 6628,
        "x": -5.1,
        "y": 26.6,
        "z": 2.5,
        "gridx": -50,
        "gridy": 47,
        "iscoverpoint": False,
        "regionId": 0
    },
    {
        "nodeid": 6629,
        "x": -5.8,
        "y": 26.6,
        "z": 2.5,
        "gridx": -49,
        "gridy": 47,
        "iscoverpoint": False,
        "regionId": 0
    },
    {
        "nodeid": 6630,
        "x": -6.5,
        "y": 26.6,
        "z": 2.5,
        "gridx": -48,
        "gridy": 47,
        "iscoverpoint": False,
        "regionId": 0
    },
    {
        "nodeid": 6631,
        "x": -5.1,
        "y": 27.3,
        "z": 2.5,
        "gridx": -50,
        "gridy": 46,
        "iscoverpoint": False,
        "regionId": 0
    },
    {
        "nodeid": 6632,
        "x": -5.8,
        "y": 27.3,
        "z": 2.5,
        "gridx": -49,
        "gridy": 46,
        "iscoverpoint": False,
        "regionId": 0
    },
    {
        "nodeid": 6633,
        "x": -6.5,
        "y": 27.3,
        "z": 2.5,
        "gridx": -48,
        "gridy": 46,
        "iscoverpoint": False,
        "regionId": 0
    },
    {
        "nodeid": 6634,
        "x": -5.1,
        "y": 28.0,
        "z": 2.5,
        "gridx": -50,
        "gridy": 45,
        "iscoverpoint": False,
        "regionId": 0
    },
    {
        "nodeid": 6635,
        "x": -5.8,
        "y": 28.0,
        "z": 2.5,
        "gridx": -49,
        "gridy": 45,
        "iscoverpoint": False,
        "regionId": 0
    },
    {
        "nodeid": 6636,
        "x": -6.5,
        "y": 28.0,
        "z": 2.5,
        "gridx": -48,
        "gridy": 45,
        "iscoverpoint": False,
        "regionId": 0
    },
    {
        "nodeid": 6637,
        "x": -28.2,
        "y": 49.7,
        "z": 4.2,
        "gridx": -17,
        "gridy": 14,
        "iscoverpoint": False,
        "regionId": 0
    },
    {
        "nodeid": 6638,
        "x": -28.2,
        "y": 50.4,
        "z": 4.2,
        "gridx": -17,
        "gridy": 13,
        "iscoverpoint": False,
        "regionId": 0
    },
    {
        "nodeid": 6639,
        "x": -20.5,
        "y": 51.1,
        "z": 5.7,
        "gridx": -28,
        "gridy": 12,
        "iscoverpoint": False,
        "regionId": 0
    },
    {
        "nodeid": 6640,
        "x": -20.5,
        "y": 50.4,
        "z": 5.7,
        "gridx": -28,
        "gridy": 13,
        "iscoverpoint": False,
        "regionId": 0
    },
    {
        "nodeid": 6641,
        "x": -20.5,
        "y": 49.7,
        "z": 5.7,
        "gridx": -28,
        "gridy": 14,
        "iscoverpoint": False,
        "regionId": 0
    },
    {
        "nodeid": 6642,
        "x": 22.9,
        "y": 48.3,
        "z": 6.35,
        "gridx": -90,
        "gridy": 16,
        "iscoverpoint": False,
        "regionId": 0
    },
    {
        "nodeid": 6643,
        "x": 23.6,
        "y": 48.3,
        "z": 6.35,
        "gridx": -91,
        "gridy": 16,
        "iscoverpoint": False,
        "regionId": 0
    },
    {
        "nodeid": 6644,
        "x": 22.9,
        "y": 47.6,
        "z": 6.35,
        "gridx": -90,
        "gridy": 17,
        "iscoverpoint": False,
        "regionId": 0
    },
    {
        "nodeid": 6645,
        "x": 23.6,
        "y": 47.6,
        "z": 6.35,
        "gridx": -91,
        "gridy": 17,
        "iscoverpoint": False,
        "regionId": 0
    }
]

ADDITIONAL_CONNECTIONS = [
    {
        "from": 6628,
        "to": 3001,
        "direction": Direction.EAST
    },
    {
        "from": 3001,
        "to": 6628,
        "direction": Direction.WEST
    },
    {
        "from": 1810,
        "to": 6628,
        "direction": Direction.NORTH
    },
    {
        "from": 6628,
        "to": 1810,
        "direction": Direction.SOUTH
    },
    {
        "from": 6629,
        "to": 1744,
        "direction": Direction.SOUTH
    },
    {
        "from": 1744,
        "to": 6629,
        "direction": Direction.NORTH
    },
    {
        "from": 6630,
        "to": 1686,
        "direction": Direction.SOUTH
    },
    {
        "from": 6630,
        "to": 1598,
        "direction": Direction.WEST
    },
    {
        "from": 6633,
        "to": 1562,
        "direction": Direction.WEST
    },
    {
        "from": 6636,
        "to": 1532,
        "direction": Direction.WEST
    },
    {
        "from": 1532,
        "to": 6636,
        "direction": Direction.EAST
    },
    {
        "from": 6636,
        "to": 1506,
        "direction": Direction.NORTH
    },
    {
        "from": 6635,
        "to": 1507,
        "direction": Direction.NORTH
    },
    {
        "from": 6634,
        "to": 1534,
        "direction": Direction.NORTH
    },
    {
        "from": 6634,
        "to": 3122,
        "direction": Direction.EAST
    },
    {
        "from": 3122,
        "to": 6634,
        "direction": Direction.WEST
    },
    {
        "from": 6631,
        "to": 3056,
        "direction": Direction.EAST
    },
    {
        "from": 3056,
        "to": 6631,
        "direction": Direction.WEST
    },
    {
        "from": 6628,
        "to": 6629,
        "direction": Direction.WEST
    },
    {
        "from": 6629,
        "to": 6628,
        "direction": Direction.EAST
    },
    {
        "from": 6629,
        "to": 6630,
        "direction": Direction.WEST
    },
    {
        "from": 6630,
        "to": 6629,
        "direction": Direction.EAST
    },
    {
        "from": 6631,
        "to": 6628,
        "direction": Direction.SOUTH
    },
    {
        "from": 6628,
        "to": 6631,
        "direction": Direction.NORTH
    },
    {
        "from": 6632,
        "to": 6629,
        "direction": Direction.SOUTH
    },
    {
        "from": 6629,
        "to": 6632,
        "direction": Direction.NORTH
    },
    {
        "from": 6633,
        "to": 6630,
        "direction": Direction.SOUTH
    },
    {
        "from": 6630,
        "to": 6633,
        "direction": Direction.NORTH
    },
    {
        "from": 6631,
        "to": 6632,
        "direction": Direction.WEST
    },
    {
        "from": 6632,
        "to": 6631,
        "direction": Direction.EAST
    },
    {
        "from": 6632,
        "to": 6633,
        "direction": Direction.WEST
    },
    {
        "from": 6633,
        "to": 6632,
        "direction": Direction.EAST
    },
    {
        "from": 6634,
        "to": 6631,
        "direction": Direction.SOUTH
    },
    {
        "from": 6631,
        "to": 6634,
        "direction": Direction.NORTH
    },
    {
        "from": 6635,
        "to": 6632,
        "direction": Direction.SOUTH
    },
    {
        "from": 6632,
        "to": 6635,
        "direction": Direction.NORTH
    },
    {
        "from": 6636,
        "to": 6633, 
        "direction": Direction.SOUTH
    },
    {
        "from": 6633,
        "to": 6636,
        "direction": Direction.NORTH
    },
    {
        "from": 6634,
        "to": 6635,
        "direction": Direction.WEST
    },
    {
        "from": 6635,
        "to": 6634,
        "direction": Direction.EAST
    },
    {
        "from": 6635,
        "to": 6636,
        "direction": Direction.WEST
    },
    {
        "from": 6636,
        "to": 6635,
        "direction": Direction.EAST
    },
    {
        "from": 6630,
        "to": 1641,
        "direction": Direction.SOUTHWEST
    },
    {
        "from": 6636,
        "to": 1501,
        "direction": Direction.NORTHWEST
    },
    {
        "from": 6628,
        "to": 6632,
        "direction": Direction.NORTHWEST
    },
    {
        "from": 6632,
        "to": 6628,
        "direction": Direction.SOUTHEAST
    },
    {
        "from": 6629,
        "to": 6631,
        "direction": Direction.NORTHEAST
    },
    {
        "from": 6631,
        "to": 6629,
        "direction": Direction.SOUTHWEST
    },
    {
        "from": 6633,
        "to": 6629,
        "direction": Direction.SOUTHEAST
    },
    {
        "from": 6629,
        "to": 6633,
        "direction": Direction.NORTHWEST
    },
    {
        "from": 6632,
        "to": 6630,
        "direction": Direction.SOUTHWEST
    },
    {
        "from": 6630,
        "to": 6632,
        "direction": Direction.NORTHEAST
    },
    {
        "from": 6635,
        "to": 6631,
        "direction": Direction.SOUTHEAST
    },
    {
        "from": 6631,
        "to": 6635,
        "direction": Direction.NORTHWEST
    },
    {
        "from": 6632,
        "to": 6634,
        "direction": Direction.NORTHEAST
    },
    {
        "from": 6634,
        "to": 6632,
        "direction": Direction.SOUTHWEST
    },
    {
        "from": 6633,
        "to": 6635,
        "direction": Direction.NORTHEAST
    },
    {
        "from": 6635,
        "to": 6633,
        "direction": Direction.SOUTHWEST
    },
    {
        "from": 6636,
        "to": 6632,
        "direction": Direction.SOUTHEAST
    },
    {
        "from": 6632,
        "to": 6636,
        "direction": Direction.NORTHWEST
    },
    {
        "from": 6637,
        "to": 227,
        "direction": Direction.SOUTH
    },
    {
        "from": 227,
        "to": 6637,
        "direction": Direction.NORTH
    },
    {
        "from": 6637,
        "to": 225,
        "direction": Direction.SOUTHWEST
    },
    {
        "from": 225,
        "to": 6637,
        "direction": Direction.NORTHEAST
    },
    {
        "from": 6637,
        "to": 208,
        "direction": Direction.WEST
    },
    {
        "from": 208,
        "to": 6637,
        "direction": Direction.EAST
    },
    {
        "from": 6637,
        "to": 206,
        "direction": Direction.NORTHWEST
    },
    {
        "from": 206,
        "to": 6637,
        "direction": Direction.SOUTHEAST
    },
    {
        "from": 6638,
        "to": 6637,
        "direction": Direction.SOUTH
    },
    {
        "from": 6637,
        "to": 6638,
        "direction": Direction.NORTH
    },
    {
        "from": 6638,
        "to": 206,
        "direction": Direction.WEST
    },
    {
        "from": 206,
        "to": 6638,
        "direction": Direction.EAST
    },
    {
        "from": 6638,
        "to": 208,
        "direction": Direction.SOUTHWEST
    },
    {
        "from": 208,
        "to": 6638,
        "direction": Direction.NORTHEAST
    },
    {
        "from": 6638,
        "to": 205,
        "direction": Direction.NORTHWEST
    },
    {
        "from": 205,
        "to": 6638,
        "direction": Direction.SOUTHEAST
    },
    {
        "from": 6638,
        "to": 226,
        "direction": Direction.NORTH
    },
    {
        "from": 226,
        "to": 6638,
        "direction": Direction.SOUTH
    },
    {
        "from": 6638,
        "to": 232,
        "direction": Direction.NORTHEAST
    },
    {
        "from": 232,
        "to": 6638,
        "direction": Direction.SOUTHWEST
    },
    {
        "from": 6637,
        "to": 928,
        "direction": Direction.EAST
    },
    {
        "from": 928,
        "to": 6637,
        "direction": Direction.WEST
    },
    {
        "from": 6638,
        "to": 928,
        "direction": Direction.SOUTHEAST
    },
    {
        "from": 928,
        "to": 6638,
        "direction": Direction.NORTHWEST
    },
    {
        "from": 6639,
        "to": 868,
        "direction": Direction.WEST
    },
    {
        "from": 868,
        "to": 6639,
        "direction": Direction.EAST
    },
    {
        "from": 6639,
        "to": 6640,
        "direction": Direction.SOUTH
    },
    {
        "from": 6640,
        "to": 6639,
        "direction": Direction.NORTH
    },
    {
        "from": 6640,
        "to": 6641,
        "direction": Direction.SOUTH
    },
    {
        "from": 6641,
        "to": 6640,
        "direction": Direction.NORTH
    },
    {
        "from": 6641,
        "to": 796,
        "direction": Direction.SOUTH
    },
    {
        "from": 6641,
        "to": 795,
        "direction": Direction.SOUTHWEST
    },
    {
        "from": 6641,
        "to": 820,
        "direction": Direction.WEST
    },
    {
        "from": 6640,
        "to": 843,
        "direction": Direction.WEST
    },
    {
        "from": 6645,
        "to": 4058,
        "direction": Direction.EAST
    },
    {
        "from": 4058,
        "to": 6645,
        "direction": Direction.WEST
    },
    {
        "from": 6643,
        "to": 3918,
        "direction": Direction.EAST
    },
    {
        "from": 3918,
        "to": 6643,
        "direction": Direction.WEST
    },
    {
        "from": 6645,
        "to": 4183,
        "direction": Direction.SOUTHEAST
    },
    {
        "from": 4183,
        "to": 6645,
        "direction": Direction.NORTHWEST
    },
    {
        "from": 6643,
        "to": 3793,
        "direction": Direction.NORTHEAST
    },
    {
        "from": 3793,
        "to": 6643,
        "direction": Direction.SOUTHWEST
    },
    {
        "from": 6645,
        "to": 4302,
        "direction": Direction.SOUTH
    },
    {
        "from": 6644,
        "to": 4407,
        "direction": Direction.SOUTH
    },
    {
        "from": 6644,
        "to": 4405,
        "direction": Direction.SOUTHWEST
    },
    {
        "from": 6644,
        "to": 4297,
        "direction": Direction.WEST
    },
    {
        "from": 6642,
        "to": 4180,
        "direction": Direction.WEST
    },
    {
        "from": 6642,
        "to": 4178,
        "direction": Direction.NORTHWEST
    },
    {
        "from": 6642,
        "to": 4055,
        "direction": Direction.NORTH
    },
    {
        "from": 6643,
        "to": 3916,
        "direction": Direction.NORTH
    },
    {
        "from": 6643,
        "to": 6645,
        "direction": Direction.SOUTH
    },
    {
        "from": 6645,
        "to": 6643,
        "direction": Direction.NORTH
    },
    {
        "from": 6645,
        "to": 6644,
        "direction": Direction.WEST
    },
    {
        "from": 6644,
        "to": 6645,
        "direction": Direction.EAST
    },
    {
        "from": 6644,
        "to": 6642,
        "direction": Direction.NORTH
    },
    {
        "from": 6642,
        "to": 6644,
        "direction": Direction.SOUTH
    },
    {
        "from": 6642,
        "to": 6643,
        "direction": Direction.EAST
    },
    {
        "from": 6643,
        "to": 6642,
        "direction": Direction.WEST
    },
    {
        "from": 6643,
        "to": 6644,
        "direction": Direction.SOUTHWEST
    },
    {
        "from": 6644,
        "to": 6643,
        "direction": Direction.NORTHEAST
    },
    {
        "from": 6642,
        "to": 6645,
        "direction": Direction.SOUTHEAST
    },
    {
        "from": 6645,
        "to": 6642,
        "direction": Direction.NORTHWEST
    }
]

REMOVE_CONNECTIONS = [
    (205, 204),
    (206, 182),
    (4183, 3354),
]

class WaypointGraph:
    def __init__(self):
        """Initialize an empty directed graph for waypoints."""
        self.graph = nx.DiGraph()
        self.regions = defaultdict(list)
        self.region2id = {}
        self.id2region = {}
        
    def load_from_json(self, json_path):
        """Load waypoints from JSON file and construct the graph."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # First pass: Create all nodes with their attributes
        for waypoint in data['nodes']:
            waypoint_id = waypoint['nodeid']
            x, y, z = float(waypoint['x']), float(waypoint['y']), float(waypoint['z'])

            self.add_waypoint(
                waypoint_id,
                x, y, z,
                waypoint['gridx'],
                waypoint['gridy'],
                waypoint['iscoverpoint'],
                waypoint['regionId']
            )
        for waypoint in ADDITIONAL_WAYPOINTS:
            self.add_waypoint(
                waypoint['nodeid'],
                waypoint['x'],
                waypoint['y'],
                waypoint['z'],
                waypoint['gridx'],
                waypoint['gridy'],
                waypoint['iscoverpoint'],
                waypoint['regionId']
            )

        # Second pass: Link edges and populate neighbors
        for waypoint in data['connections']:
            from_node = waypoint['from']
            to_node = waypoint['to']
            direction = Direction(waypoint['direction'])
            self.add_connection(from_node, to_node, direction)
            
        for connection in ADDITIONAL_CONNECTIONS:
            from_node = connection['from']
            to_node = connection['to']
            direction = Direction(connection['direction'])
            self.add_connection(from_node, to_node, direction)

        for region in data['regions']:
            self.region2id[region['name']] = Region(region['id'])
            self.id2region[Region(region['id'])] = region['name']
            
        for connection in REMOVE_CONNECTIONS:
            self.remove_connection(connection[0], connection[1])

    def add_waypoint(self, waypoint_id, x, y, z, grid_x, grid_y, is_cover_point, region_type):
        """Add a new waypoint to the graph."""
        self.graph.add_node(waypoint_id, 
                            id=waypoint_id, 
                            pos=Vec3(x, y, z), 
                            grid_x=int(grid_x), 
                            grid_y=int(grid_y), 
                            is_cover_point=bool(is_cover_point), 
                            region_type=Region(int(region_type)), 
                            neighbor_ids={}
        )
        if region_type != 0:
            self.regions[Region(region_type)].append(waypoint_id)

    def add_connection(self, from_id, to_id, direction):
        """Add a new connection between two waypoints."""
        if to_id in self.graph.nodes:
            self.graph.add_edge(from_id, to_id, direction=direction)
            self.graph.nodes[from_id]['neighbor_ids'][direction] = to_id

    def remove_waypoint(self, waypoint_id):
        """Remove a waypoint from the graph and all connections associated with it."""
        # Remove waypoint from regions if it exists
        region_type = self.graph.nodes[waypoint_id].get('region_type')
        if region_type and region_type in self.regions:
            self.regions[region_type].remove(waypoint_id)

        # Remove from neighbor_ids of all connected nodes
        for node in list(self.graph.predecessors(waypoint_id)):
            node_data = self.graph.nodes[node]
            node_data['neighbor_ids'] = {
                direction: neighbor 
                for direction, neighbor in node_data['neighbor_ids'].items() 
                if neighbor != waypoint_id
            }
            self.graph.remove_edge(node, waypoint_id)

        for node in list(self.graph.successors(waypoint_id)):
            self.graph.remove_edge(waypoint_id, node)

        # Remove the node and all its connections
        self.graph.remove_node(waypoint_id)

    def remove_connection(self, from_id, to_id):
        """Remove a connection between two waypoints."""
        # First find the direction of this connection
        from_node = self.graph.nodes[from_id]
        direction_to_remove = None
        
        # Find which direction maps to the target waypoint
        for direction, neighbor_id in from_node['neighbor_ids'].items():
            if neighbor_id == to_id:
                direction_to_remove = direction
                break
        
        if direction_to_remove is not None:
            del from_node['neighbor_ids'][direction_to_remove]
        self.graph.remove_edge(from_id, to_id)

    def get_waypoint_by_id(self, node_id, return_pos=False):
        """Get the waypoint by its ID."""
        waypoint = self.graph.nodes[node_id]
        if return_pos:
            return waypoint['pos']
        return waypoint
    
    def is_neighbor_valid(self, node_id, direction):
        """Check if a waypoint has a valid neighbor in a given direction."""
        return direction in self.graph.nodes[node_id]['neighbor_ids']

    def get_neighbors(self, node_id, return_id=False, return_pos=False):
        """Get all neighboring waypoint IDs."""
        assert not (return_id and return_pos), "Cannot return both ID and position"
        if return_id:
            return list(self.graph.neighbors(node_id))
        elif return_pos:
            return [self.graph.nodes[node_id]['pos'] for node_id in self.graph.neighbors(node_id)]
        else:
            return [self.graph.nodes[node_id] for node_id in self.graph.neighbors(node_id)]

    def get_neighbor_by_direction(self, node_id, direction, return_id=False, return_pos=False):
        """Get the neighboring waypoint ID in a given direction."""
        assert not (return_id and return_pos), "Cannot return both ID and position"
        assert isinstance(direction, Direction), "Direction must be an instance of Direction Enum"
        if not self.is_neighbor_valid(node_id, direction):
            raise ValueError(f"Direction {direction} not found for waypoint {node_id}")
        
        neighbor_waypoint_id = self.graph.nodes[node_id]['neighbor_ids'][direction]
        neighbor_waypoint = self.graph.nodes[neighbor_waypoint_id]

        if return_id:
            return neighbor_waypoint_id
        elif return_pos:
            return neighbor_waypoint['pos']
        return neighbor_waypoint
    
    def get_nearest_waypoint(self, position, return_id=False, return_pos=False):
        """Find the closest waypoint to a given position."""
        assert not (return_id and return_pos), "Cannot return both ID and position"

        min_distance = float('inf')
        nearest_id = None
        
        for node_id, attr in self.graph.nodes(data=True):
            if not attr:  # This shouldn't happen if nodes were added correctly
                print(f"Warning: Node {node_id} has no attributes!")
                continue
            waypoint_pos = attr['pos']
            distance = vec_distance(position, waypoint_pos)
            
            if distance < min_distance:
                min_distance = distance
                nearest_id = node_id
        
        if return_id:
            return nearest_id
        elif return_pos:
            return self.get_position(nearest_id)
        return self.graph.nodes[nearest_id]
    
    def get_random_waypoint(self, return_id=False, return_pos=False):
        """Get a random waypoint from the graph."""
        assert not (return_id and return_pos), "Cannot return both ID and position"
        waypoint_id = random.choice(list(self.graph.nodes))
        waypoint = self.graph.nodes[waypoint_id]
        if return_id:
            return waypoint_id
        elif return_pos:
            return waypoint['pos']
        return waypoint
    
    def get_random_waypoint_in_region(self, region_type: Region, return_id=False, return_pos=False):
        """Get a random waypoint in a specific region."""
        assert not (return_id and return_pos), "Cannot return both ID and position"
        waypoint_id = random.choice(self.regions[region_type])
        waypoint = self.graph.nodes[waypoint_id]
        if return_id:
            return waypoint_id
        elif return_pos:
            return waypoint['pos']
        return waypoint
    
    def get_region_of_waypoint(self, waypoint_id):
        """Get the region of a waypoint."""
        return self.graph.nodes[waypoint_id]['region_type']
    
    def find_path(self, start_id, end_id):
        """Find shortest path between two waypoints."""
        path = nx.shortest_path(self.graph, start_id, end_id)
        return path
        
    def get_interpolated_waypoint_path(self, positions, return_actions=False):
        """
        Get the interpolated waypoint path between two positions.
        """
        full_path = []
        full_actions = []
        prev_waypoint_id = None
        for i in range(1, len(positions)):
            start_pos = Vec3(*positions[i-1])
            end_pos = Vec3(*positions[i])
            start_id = self.get_nearest_waypoint(start_pos, return_id=True) if prev_waypoint_id is None else prev_waypoint_id
            end_id = self.get_nearest_waypoint(end_pos, return_id=True)
            
            path = self.find_path(start_id, end_id)
            prev_waypoint_id = end_id
            full_path.extend(path if i == 1 or start_id == end_id else path[1:])
        if return_actions:
            for i in range(1, len(full_path)):
                start_id = full_path[i-1]
                end_id = full_path[i]
                action = STOP_ACTION_INDEX if start_id == end_id else self.graph.edges[start_id, end_id]['direction'].value
                if start_id != end_id:
                    direction = self.graph.edges[start_id, end_id]['direction']
                full_actions.append(action)
        return full_path, full_actions

    def save_visualization(self, output_path, highlight_ids=None):
        """
        Save a 2D visualization of the waypoint graph using Plotly.
        Args:
            output_path (str): Path where the HTML file will be saved
            highlight_ids (list, optional): List of waypoint IDs to highlight in yellow
        """
        # Get positions for all nodes
        x_coords = []
        y_coords = []
        node_ids = []
        node_colors = []
        
        highlight_ids = highlight_ids or []
        
        for node, attr in self.graph.nodes(data=True):
            if not attr:
                print(f"Warning: Node {node} has no attributes!")
                continue
            x_coords.append(attr['pos'].getX())
            y_coords.append(attr['pos'].getY())
            node_ids.append(node)
            node_colors.append('yellow' if node in highlight_ids else 'green' if attr['region_type'] != Region.NONE else 'blue')
        
        # Create scatter plot for nodes with updated hover text
        node_trace = go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers',
            hoverinfo='text',
            text=[f'Node ID: {node_id}<br>'
                  f'Position: ({attr["pos"].getX():.1f}, {attr["pos"].getY():.1f}, {attr["pos"].getZ():.1f})<br>'
                  f'Grid: ({attr["grid_x"]}, {attr["grid_y"]})<br>'
                  f'Cover Point: {attr["is_cover_point"]}<br>'
                  f'Region: {attr["region_type"] if attr["region_type"] != 0 else "None"}<br>'
                  f'Neighbors: {dict([(str(k), v) for k,v in attr["neighbor_ids"].items()])}'
                  for node_id, attr in self.graph.nodes(data=True)],
            marker=dict(
                size=5,
                color=node_colors,
            )
        )
        
        # Create separate traces for bidirectional and single-direction edges with hover text
        bidirectional_x = []
        bidirectional_y = []
        bidirectional_text = []
        directed_x = []
        directed_y = []
        directed_text = []
        
        # Find bidirectional edges
        bidirectional_edges = set()
        for edge in self.graph.edges():
            if (edge[1], edge[0]) in self.graph.edges():
                sorted_edge = tuple(sorted([edge[0], edge[1]]))
                bidirectional_edges.add(sorted_edge)
        
        # Create edge traces with hover text
        for edge in self.graph.edges():
            x0, y0 = self.graph.nodes[edge[0]]['pos'].getX(), self.graph.nodes[edge[0]]['pos'].getY()
            x1, y1 = self.graph.nodes[edge[1]]['pos'].getX(), self.graph.nodes[edge[1]]['pos'].getY()
            direction = self.graph.edges[edge[0], edge[1]]['direction']
            hover_text = f'From: {edge[0]}<br>To: {edge[1]}<br>Direction: {direction}'
            
            sorted_edge = tuple(sorted([edge[0], edge[1]]))
            if sorted_edge in bidirectional_edges:
                bidirectional_x.extend([x0, x1, None])
                bidirectional_y.extend([y0, y1, None])
                bidirectional_text.extend([hover_text, hover_text, None])
            else:
                directed_x.extend([x0, x1, None])
                directed_y.extend([y0, y1, None])
                directed_text.extend([hover_text, hover_text, None])
        
        # Create bidirectional edge trace with hover text
        bidirectional_trace = go.Scatter(
            x=bidirectional_x,
            y=bidirectional_y,
            mode='lines',
            line=dict(width=0.5, color='#888'),
            hoverinfo='text',
            text=bidirectional_text
        )
        
        # Create directed edge trace with hover text
        directed_trace = go.Scatter(
            x=directed_x,
            y=directed_y,
            mode='lines',
            line=dict(width=0.5, color='red'),
            hoverinfo='text',
            text=directed_text
        )
        
        # Create the figure
        fig = go.Figure(
            data=[bidirectional_trace, directed_trace, node_trace],
            layout=go.Layout(
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0, l=0, r=0, t=0),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        
        # Add arrows for directed edges using annotations
        for edge in self.graph.edges():
            if tuple(sorted([edge[0], edge[1]])) not in bidirectional_edges:
                x0, y0 = self.graph.nodes[edge[0]]['pos'].getX(), self.graph.nodes[edge[0]]['pos'].getY()
                x1, y1 = self.graph.nodes[edge[1]]['pos'].getX(), self.graph.nodes[edge[1]]['pos'].getY()
                
                # Calculate the position for the arrow (80% along the edge)
                ax = x0 + 0.8 * (x1 - x0)
                ay = y0 + 0.8 * (y1 - y0)
                
                fig.add_annotation(
                    x=ax,
                    y=ay,
                    ax=x1,
                    ay=y1,
                    xref='x',
                    yref='y',
                    axref='x',
                    ayref='y',
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1,
                    arrowcolor='red'
                )
        
        # Save as HTML file
        fig.write_html(output_path)

    def export_to_json(self, json_path):
        """Export the graph to JSON file in the same format as the input."""
        output_data = {
            "nodes": [],
            "connections": [],
            "regions": []
        }

        # Export nodes
        for node_id, attr in self.graph.nodes(data=True):
            node_data = {
                "nodeid": node_id,
                "x": float(attr['pos'].getX()),  # Convert back to original coordinate system
                "y": float(attr['pos'].getY()),
                "z": float(attr['pos'].getZ()),
                "gridx": int(attr['grid_x']),
                "gridy": int(attr['grid_y']),
                "iscoverpoint": bool(attr['is_cover_point']),
                "regionId": attr['region_type'].value
            }
            output_data["nodes"].append(node_data)

        # Export connections
        for from_node, to_node, edge_data in self.graph.edges(data=True):
            connection = {
                "from": from_node,
                "to": to_node,
                "direction": edge_data['direction'].value
            }
            output_data["connections"].append(connection)

        # Export regions
        for region_id, region_name in self.id2region.items():
            region_data = {
                "id": region_id.value,
                "name": region_name
            }
            output_data["regions"].append(region_data)

        # Write to file
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=4)

    def verify_graph_integrity(self, verbose=False):
        """
        Verify the integrity of the graph by checking connectivity and neighbor consistency.
        
        Args:
            verbose (bool): If True, print detailed information about any issues found
        
        Returns:
            bool: True if graph is strongly connected and has no neighbor inconsistencies
        """
        # Check strong connectivity
        is_connected = nx.is_strongly_connected(self.graph)
        if not is_connected and verbose:
            components = list(nx.strongly_connected_components(self.graph))
            print(f"Warning: Graph is not strongly connected!")
            print(f"Number of strongly connected components: {len(components)}")
            print(f"Component sizes: {[len(c) for c in components]}")
            
            # Print nodes in small components
            small_components = [c for c in components if len(c) < 5]
            if small_components:
                print("\nNodes in small components (size < 5):")
                for comp in small_components:
                    print(f"Component size {len(comp)}: Nodes {list(comp)}")
        
        # Check neighbor consistency
        inconsistencies = []
        for node_id in self.graph.nodes:
            node_data = self.graph.nodes[node_id]
            
            # Check if each neighbor_id in the dictionary corresponds to an actual edge
            for direction, neighbor_id in node_data['neighbor_ids'].items():
                if not self.graph.has_edge(node_id, neighbor_id):
                    inconsistencies.append(
                        f"Node {node_id} claims {neighbor_id} as neighbor in direction {direction}, "
                        "but no corresponding edge exists"
                    )
            
            # Check if each edge has a corresponding neighbor_id entry
            for _, neighbor_id, edge_data in self.graph.edges(node_id, data=True):
                direction = edge_data['direction']
                if direction not in node_data['neighbor_ids'] or node_data['neighbor_ids'][direction] != neighbor_id:
                    inconsistencies.append(
                        f"Edge exists from {node_id} to {neighbor_id} in direction {direction}, "
                        "but no corresponding neighbor_id entry found"
                    )
        
        if inconsistencies and verbose:
            print("\nFound neighbor inconsistencies:")
            for msg in inconsistencies:
                print(msg)
        
        if is_connected and not inconsistencies and verbose:
            print("Graph integrity verified: All checks passed!")
        
        return is_connected and not inconsistencies

if __name__ == '__main__':
    graph = WaypointGraph()
    graph.load_from_json(WAYPOINT_DATA_PATH)
    
    # Verify graph integrity with verbose output
    graph.verify_graph_integrity(verbose=True)
    
    # Save visualization
    # highlight_ids = [1524, 1493, 1471, 1449, 1430, 1411, 1389, 1365, 1342, 1317, 1291, 1257, 1215, 1181, 1222, 1264, 1294, 1319, 1346, 1371, 1372, 1373, 1350, 1375, 1351, 1326, 1298, 1267, 1223, 1180, 1175, 1213, 1253, 1211, 1171, 1133, 1171, 1211, 1211, 1211, 1211, 1211, 1211, 1211, 1211, 1211, 1211, 1211, 1211, 1211, 1211, 1211, 1211, 1211, 1211, 1211, 1211, 1211, 1211, 1211, 1211, 1211, 1211, 1211, 1211, 1211, 1211, 1211, 1211, 1211, 1211, 1211, 1211, 1211, 1211, 1211, 1211, 1211, 1211, 1211, 1211, 1211, 1211, 1211, 1211, 1211, 1211, 1211, 1211, 1211, 1211]
    # graph.save_visualization(WAYPOINT_DATA_PATH.replace('.json', '.html'), highlight_ids=highlight_ids)
    # graph.save_visualization(WAYPOINT_DATA_PATH.replace('.json', '.html'))
    graph.save_visualization(WAYPOINT_DATA_PATH.replace('.json', '_manual.html'))

