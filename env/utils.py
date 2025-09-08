from panda3d.core import Vec3
from enum import Enum, auto
from typing import List, Tuple, Dict
import numpy as np
from .config import COORDINATE_SCALE

class AgentStats:
    def __init__(self):
        self.total_distance: float = 0.0
        self.total_xy_distance: float = 0.0
        self.total_jumps: int = 0
        self.total_reward: float = 0.0
        self.stuck_count: int = 0
        self.decision_ticks: List[Tuple[Direction, int]] = []

    @property
    def total_decisions(self):
        return len(self.decision_ticks)
    
    @property
    def total_cardinal_decisions(self):
        return sum(1 for direction, _ in self.decision_ticks if direction in [Direction.N, Direction.E, Direction.S, Direction.W])
    
    @property
    def total_diagonal_decisions(self):
        return sum(1 for direction, _ in self.decision_ticks if direction in [Direction.NE, Direction.NW, Direction.SE, Direction.SW])
    
    @property
    def avg_decision_tick(self):
        return sum(tick for _, tick in self.decision_ticks) / self.total_decisions
    
    @property
    def avg_decision_tick_cardinal(self):
        return sum(tick for direction, tick in self.decision_ticks if direction in [Direction.N, Direction.E, Direction.S, Direction.W]) / self.total_cardinal_decisions
    
    @property
    def avg_decision_tick_diagonal(self):
        return sum(tick for direction, tick in self.decision_ticks if direction in [Direction.NE, Direction.NW, Direction.SE, Direction.SW]) / self.total_diagonal_decisions

    @property
    def max_cardinal_tick(self):
        cardinal_ticks = [tick for direction, tick in self.decision_ticks 
                         if direction in [Direction.N, Direction.E, Direction.S, Direction.W]]
        return max(cardinal_ticks) if cardinal_ticks else 0
    
    @property
    def min_cardinal_tick(self):
        cardinal_ticks = [tick for direction, tick in self.decision_ticks 
                         if direction in [Direction.N, Direction.E, Direction.S, Direction.W]]
        return min(cardinal_ticks) if cardinal_ticks else 0
    
    @property
    def max_diagonal_tick(self):
        diagonal_ticks = [tick for direction, tick in self.decision_ticks 
                         if direction in [Direction.NE, Direction.NW, Direction.SE, Direction.SW]]
        return max(diagonal_ticks) if diagonal_ticks else 0
    
    @property
    def min_diagonal_tick(self):
        diagonal_ticks = [tick for direction, tick in self.decision_ticks 
                         if direction in [Direction.NE, Direction.NW, Direction.SE, Direction.SW]]
        return min(diagonal_ticks) if diagonal_ticks else 0

    def reset(self):
        self.total_distance = 0.0
        self.total_xy_distance = 0.0
        self.total_jumps = 0
        self.total_reward = 0.0
        self.stuck_count = 0
        self.decision_ticks = []

    def to_dict(self) -> Dict[str, float]:
        return {
            "total_distance": self.total_distance,
            "total_cardinal_decisions": self.total_cardinal_decisions,
            "total_diagonal_decisions": self.total_diagonal_decisions,
            "total_jumps": self.total_jumps,
            "total_reward": self.total_reward,
            "stuck_count": self.stuck_count
        }


class Region(Enum):
    NONE = 0
    T_SPAWN = 1
    CT_SPAWN = 2
    A_BOMBSITE = 3
    B_BOMBSITE = 4

class Team(Enum):
    T = 0
    CT = 1

# class Direction(Enum):
#     NORTH = N = FORWARD = PLUS_Y = 0
#     EAST = E = RIGHT = PLUS_X = 1
#     SOUTH = S = BACK = MINUS_Y = 2
#     WEST = W = LEFT = MINUS_X = 3
#     NORTHEAST = NE = FORWARD_RIGHT = PLUS_X_PLUS_Y = 4
#     NORTHWEST = NW = FORWARD_LEFT = PLUS_X_MINUS_Y = 5
#     SOUTHEAST = SE = BACK_RIGHT = MINUS_X_PLUS_Y = 6
#     SOUTHWEST = SW = BACK_LEFT = MINUS_X_MINUS_Y = 7

class Direction(Enum):
    SOUTH = S = BACK = MINUS_Y = 0
    WEST = W = LEFT = MINUS_X = 1
    NORTH = N = FORWARD = PLUS_Y = 2
    EAST = E = RIGHT = PLUS_X = 3
    SOUTHWEST = SW = BACK_LEFT = MINUS_X_MINUS_Y = 4
    NORTHWEST = NW = FORWARD_LEFT = PLUS_X_MINUS_Y = 5
    SOUTHEAST = SE = BACK_RIGHT = MINUS_X_PLUS_Y = 6
    NORTHEAST = NE = FORWARD_RIGHT = PLUS_X_PLUS_Y = 7


# public enum Direction
# {
#     Z_Positive,                // Index 0: Vector3.forward (Z Positive)
#     X_Positive,                // Index 1: Vector3.right (X Positive)
#     Z_Negative,                // Index 2: Vector3.back (Z Negative)
#     X_Negative,                // Index 3: Vector3.left (X Negative)
#     X_Positive_Z_Positive,     // Index 4: Vector3.forward + Vector3.right (Z Positive + X Positive)
#     X_Negative_Z_Positive,     // Index 5: Vector3.forward + Vector3.left (Z Positive + X Negative)
#     X_Positive_Z_Negative,     // Index 6: Vector3.back + Vector3.right (Z Negative + X Positive)
#     X_Negative_Z_Negative      // Index 7: Vector3.back + Vector3.left (Z Negative + X Negative)
# }

class BombStatus(Enum):
    Carried = 0
    Dropped = 1
    Planted = 2
    Defused = 3
    Detonated = 4

class WinReason(Enum):
    TimeOut = 0
    TerroristEliminated = 1
    CounterTerroristEliminated = 2
    BombDetonated = 3
    BombDefused = 4

class Weapon(Enum):
    AK_47 = 0
    AUG = 1
    AWP = 2
    CZ75_Auto = 3
    Desert_Eagle = 4
    Dual_Berettas = 5
    FAMAS = 6
    Five_SeveN = 7
    G3SG1 = 8
    Galil_AR = 9
    Glock_18 = 10
    M249 = 11
    M4A1 = 12
    M4A4 = 13
    MAC_10 = 14
    MAG_7 = 15
    MP5_SD = 16
    MP7 = 17
    MP9 = 18
    Negev = 19
    Nova = 20
    P2000 = 21
    P250 = 22
    P90 = 23
    PP_Bizon = 24
    R8_Revolver = 25
    SCAR_20 = 26
    SG_553 = 27
    SSG_08 = 28
    Sawed_Off = 29
    Tec_9 = 30
    UMP_45 = 31
    USP_S = 32
    XM1014 = 33

def vec_distance(vec1, vec2):
    return (vec1 - vec2).length()

def get_opposite_team(team: Team) -> Team:
    return Team.CT if team == Team.T else Team.T

def rotate_points(points, angle_deg, axis):
    """
    Rotates a set of 3D points around the specified axis by a given angle.
    
    Parameters:
    - points: (N, 3) NumPy array of 3D points
    - angle_deg: Rotation angle in degrees
    - axis: Axis of rotation (0 for X, 1 for Y, 2 for Z)
    
    Returns:
    - Rotated points as a NumPy array
    """
    # Convert angle to radians
    angle_rad = np.radians(angle_deg)
    
    # Define rotation matrices
    if axis == 0:  # Rotation around X-axis
        R = np.array([[1, 0, 0],
                      [0, np.cos(angle_rad), -np.sin(angle_rad)],
                      [0, np.sin(angle_rad), np.cos(angle_rad)]])
    
    elif axis == 1:  # Rotation around Y-axis
        R = np.array([[np.cos(angle_rad), 0, np.sin(angle_rad)],
                      [0, 1, 0],
                      [-np.sin(angle_rad), 0, np.cos(angle_rad)]])
    
    elif axis == 2:  # Rotation around Z-axis
        R = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                      [np.sin(angle_rad), np.cos(angle_rad), 0],
                      [0, 0, 1]])
    
    else:
        raise ValueError("Axis must be 0 (X), 1 (Y), or 2 (Z).")

    # Apply rotation to all points
    rotated_points = np.dot(points, R.T)
    
    return rotated_points


def transform_csgo_to_panda3d(points):
    """
    Transforms points from CS:GO coordinate space to Panda3D coordinate space.
    
    The transformation applies:
    1. Scaling by COORDINATE_SCALE (0.01905)
    2. Translation by [0.0, 0.0, 3.1]
    
    Parameters:
        points (np.ndarray): Array of points in CS:GO space, shape (..., 3)
        
    Returns:
        np.ndarray: Transformed points in Panda3D space, same shape as input
    """
    return points * COORDINATE_SCALE + np.array([0.0, 0.0, 3.1])

def transform_panda3d_to_csgo(points):
    return (points - np.array([0.0, 0.0, 3.1])) / COORDINATE_SCALE

# def tramsform_to_minimap(points, map_size):
#     csgo_points = transform_panda3d_to_csgo(points)
#     scale = 4.4
#     pos_x = -2476
#     pos_y = 3239
#     x = csgo_points[0]
#     y = csgo_points[1]
#     x = (x - pos_x) / scale
#     y = (y - pos_y) / scale
#     x = (x / map_size) * 2 - 1
#     y = (y / map_size) * 2 - 1
#     return (x, y)


def tramsform_to_minimap(points, map_size):
    csgo_points = transform_panda3d_to_csgo(points)
    scale = 4.4
    pos_x = -2476
    pos_y = 3239
    x = csgo_points[0]
    y = csgo_points[1]
    x = (x - pos_x) / scale
    y = (pos_y - y) / scale
    x = (x / map_size) * 2 - 1
    y = (y / map_size) * 2 - 1
    return (x, -y)

def bomb_status_to_onehot(status: BombStatus) -> np.ndarray:
    """Convert BombStatus enum to one-hot encoded numpy array."""
    onehot = np.zeros(len(BombStatus), dtype=np.float32)
    onehot[status.value] = 1.0
    return onehot

if __name__ == "__main__":
    print(Team.T)
    print(type(Team.T))
    print(Team["T"].value)

    print(Vec3.forward())
    print(Vec3.back())
    print(Vec3.right())
    print(Vec3.left())
    print(Vec3.up())
    print(Vec3.down())