import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from multiprocessing import Pool, cpu_count
from functools import partial
import time
from sklearn.neighbors import NearestNeighbors
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Union, Any


MAP_NAMES = [
    'de_ancient', 'de_dust2', 'de_inferno', 'de_train', 'de_mirage',
    'de_nuke', 'de_overpass', 'de_vertigo'
]

WEAPON_NAMES = [
    'AK-47', 'AUG', 'AWP', 'CZ75 Auto', 'Desert Eagle',
    'Dual Berettas', 'FAMAS', 'Five-SeveN', 'G3SG1', 'Galil AR',
    'Glock-18', 'M249', 'M4A1', 'M4A4', 'MAC-10', 'MAG-7', 'MP5-SD', 'MP7', 'MP9', 'Negev',
    'Nova', 'P2000', 'P250', 'P90', 'PP-Bizon', 'R8 Revolver', 'SCAR-20', 'SG 553',
    'SSG 08', 'Sawed-Off', 'Tec-9', 'UMP-45', 'USP-S', 'XM1014'
]

# Add hit group names and priority mapping
HIT_GROUP_NAMES = ['Head', 'Neck', 'Stomach', 'Chest', 'Arm', 'Leg']
HIT_GROUP_PRIORITY = {
    'Head': 0,
    'Neck': 1,
    'Stomach': 2,
    'Chest': 3,
    'Arm': 4,
    'Leg': 5,
}

@dataclass
class DataPoint:
    """Represents a single data point for damage prediction."""
    map_name: str
    attacker_x: float
    attacker_y: float
    attacker_z: float
    attacker_view_x: float
    victim_x: float
    victim_y: float
    victim_z: float
    victim_view_x: float
    attacker_hp: float
    attacker_weapon: str
    victim_has_helmet: bool
    victim_has_armor: bool
    has_damage: bool
    total_damage: float
    hit_group: Optional[str] = None

    @classmethod
    def from_tuple(cls, data_tuple: tuple) -> 'DataPoint':
        return cls(*data_tuple)
    

class DamageOutcomeDataset(Dataset):
    """Dataset for CS:GO damage outcome prediction.
    
    Each data point is a tuple containing:
        map_name (str): Name of the CS:GO map (e.g., 'de_overpass')
        attacker_x (float): X coordinate of attacking player
        attacker_y (float): Y coordinate of attacking player
        attacker_z (float): Z coordinate (height) of attacking player
        attacker_view_x (float): Horizontal view angle of attacker in degrees (0-360)
        victim_x (float): X coordinate of victim player
        victim_y (float): Y coordinate of victim player
        victim_z (float): Z coordinate (height) of victim player
        victim_view_x (float): Horizontal view angle of victim in degrees (0-360)
        attacker_hp (float): Health points of attacker (0-100)
        attacker_weapon (str): Weapon being held by attacker (e.g., 'P250', 'AK47')
        victim_has_helmet (bool): Whether victim has helmet protection
        victim_has_armor (bool): Whether victim has body armor

        has_damage (bool): Whether any damage was dealt in this interaction
        total_damage (float): Total damage dealt in this interaction (0 if no damage)
        hit_group (str): Hit group of the damage (e.g., 'Head', 'Stomach')
    """
    def __init__(self, data_files, positive_samples, num_workers=5, positive_only=True):
        self.positive_samples = positive_samples
        self.positive_only = positive_only
        self.files = data_files
        
        self.num_workers = num_workers
        self.current_file_idx = 0
        self.current_data = []
        
        self.map_names = MAP_NAMES
        self.weapons = WEAPON_NAMES
        self.hit_group_names = HIT_GROUP_NAMES
        
        self.map_to_idx = {name: idx for idx, name in enumerate(self.map_names)}
        self.weapon_to_idx = {weapon: idx for idx, weapon in enumerate(self.weapons)}
        self.hit_group_to_idx = {group: idx for idx, group in enumerate(self.hit_group_names)}

        self.hit_group_class_weights = self.get_hit_group_class_weights()
        
    def __len__(self):
        return len(self.current_data)
    
    def __getitem__(self, idx):
        data_point = self.current_data[idx]
        
        map_one_hot = torch.zeros(len(self.map_names))
        map_one_hot[self.map_to_idx[data_point.map_name]] = 1
        
        # Extract position data
        attacker_x = float(data_point.attacker_x)
        attacker_y = float(data_point.attacker_y)
        attacker_z = float(data_point.attacker_z)
        attacker_view_x = float(data_point.attacker_view_x)
        victim_x = float(data_point.victim_x)
        victim_y = float(data_point.victim_y)
        victim_z = float(data_point.victim_z)
        
        # Calculate distance between players
        distance = torch.sqrt(
            torch.tensor((attacker_x - victim_x)**2 + 
                         (attacker_y - victim_y)**2 + 
                         (attacker_z - victim_z)**2)
        )
        
        # Calculate relative angle
        dx = victim_x - attacker_x
        dy = victim_y - attacker_y
        
        attacker_view_rad = torch.tensor(attacker_view_x * np.pi / 180.0)
        vector_angle = torch.atan2(torch.tensor(dy), torch.tensor(dx))
        relative_angle = torch.remainder(vector_angle - attacker_view_rad + np.pi, 2 * np.pi) - np.pi
        normalized_relative_angle = relative_angle / np.pi
        
        # Combine position and angle features
        coords_and_angles = torch.tensor([
            attacker_x, attacker_y, attacker_z,  # attacker xyz
            float(data_point.attacker_view_x) / 360.0,  # normalized attacker_view_x
            victim_x, victim_y, victim_z,  # victim xyz
            float(data_point.victim_view_x) / 360.0,  # normalized victim_view_x
            float(data_point.attacker_hp) / 100.0,  # normalized attacker_hp
            distance.item(),  # distance between attacker and victim
            normalized_relative_angle.item(),  # relative angle between attacker's view and victim
        ], dtype=torch.float32)
        
        # One-hot encode weapon
        weapon_one_hot = torch.zeros(len(self.weapons))
        weapon_one_hot[self.weapon_to_idx[data_point.attacker_weapon]] = 1
        
        # Armor features
        armor_features = torch.tensor([
            float(bool(data_point.victim_has_helmet)),  # victim_has_helmet
            float(bool(data_point.victim_has_armor)),  # victim_has_armor
        ], dtype=torch.float32)
        
        hit_group_idx = torch.tensor(-1, dtype=torch.long)
        if data_point.hit_group is not None:
            hit_group_idx = torch.tensor(self.hit_group_to_idx[data_point.hit_group], dtype=torch.long)
        
        # Target values
        damage_indicator = torch.tensor([float(bool(data_point.has_damage))], dtype=torch.float32)
        damage_value = torch.tensor([float(data_point.total_damage)], dtype=torch.float32)
        
        return {
            'map': map_one_hot,
            'coords_and_angles': coords_and_angles,
            'weapon': weapon_one_hot,
            'armor_features': armor_features,
            'damage_indicator': damage_indicator,
            'damage_value': damage_value,
            'hit_group': hit_group_idx
        }
    
    def load_next_chunk(self):
        """Load files into memory until we have enough negative samples to match positive samples.
        """
        positive_samples_dp = [DataPoint.from_tuple(data_point) for data_point in self.positive_samples]
        if self.positive_only:
            self.current_data = positive_samples_dp
            return
        
        negative_samples = []
        num_negative_samples = len(self.positive_samples)

        
        if self.current_file_idx >= len(self.files):
            self.current_file_idx = 0
            np.random.shuffle(self.files)
        
        processed_files = 0
        with Pool(processes=self.num_workers) as pool:
            file_iterator = pool.imap(
                self._process_single_file, 
                self.files[self.current_file_idx:]
            )
            
            pbar = tqdm(desc="Collecting negative samples", total=num_negative_samples)
            last_count = 0
            
            for file_data in file_iterator:
                processed_files += 1
                
                # Extract negative samples from this file
                file_negative_samples = [item for item in file_data if not item.has_damage]
                negative_samples.extend(file_negative_samples)
                
                # Update progress bar to show current count
                current_count = min(len(negative_samples), num_negative_samples)
                pbar.update(current_count - last_count)
                last_count = current_count
                
                # If we have enough negative samples, stop processing
                if len(negative_samples) >= num_negative_samples:
                    negative_samples = negative_samples[:num_negative_samples]
                    pbar.update(num_negative_samples - last_count)  # Ensure we reach 100%
                    break
            pbar.close()
        
        self.current_file_idx += processed_files
        self.current_data = positive_samples_dp + negative_samples

    
    @staticmethod
    def _process_single_file(file_path):
        """Process a single data file and extract game data points."""
        local_data = []
        traj = np.load(file_path, allow_pickle=True)
        traj_len = max(traj['player_seq_len'])
        for frame_idx in range(traj_len):
            player_trajectory = traj['player_trajectory'][:, frame_idx, :]
            player_hp_timeseries = traj['player_hp_timeseries'][:, frame_idx]
            player_view_x = traj['player_view_x'][:, frame_idx]
            player_armor = traj['player_armor']
            player_helmet = traj['player_helmet']
            player_weapons = traj['player_weapons_timeseries'][:, frame_idx]
            map_name = traj['map_name'].item()
            damage_outcome = traj['damage_outcomes'].item().get(frame_idx, [])
            
            # Iterate over T players (0-4) and CT players (5-9) both ways
            for t_idx in range(5):
                for ct_idx in range(5, 10):
                    for attacker_idx, victim_idx in [(t_idx, ct_idx), (ct_idx, t_idx)]:
                        if frame_idx > traj['player_seq_len'][attacker_idx] or frame_idx > traj['player_seq_len'][victim_idx]:
                            continue
                        if player_weapons[attacker_idx] == 'None':
                            continue
                        
                        attacker_x = player_trajectory[attacker_idx, 0]
                        attacker_y = player_trajectory[attacker_idx, 1]
                        attacker_z = player_trajectory[attacker_idx, 2]
                        attacker_view_x = player_view_x[attacker_idx]
                        victim_x = player_trajectory[victim_idx, 0]
                        victim_y = player_trajectory[victim_idx, 1]
                        victim_z = player_trajectory[victim_idx, 2]
                        victim_view_x = player_view_x[victim_idx]
                        attacker_hp = player_hp_timeseries[attacker_idx]
                        attacker_weapon = player_weapons[attacker_idx]
                        victim_has_helmet = player_helmet[victim_idx]
                        victim_has_armor = player_armor[victim_idx] > 0
                        
                        # Process damage information
                        total_damage = 0
                        hit_group = None  # Default hit group
                        hit_group_priority = float('inf')  # Start with lowest priority
                        
                        for damage_event in damage_outcome:
                            aid, vid, d, hg = damage_event
                            if aid == attacker_idx and vid == victim_idx:
                                total_damage += d
                                # Update hit group based on priority
                                current_priority = HIT_GROUP_PRIORITY.get(hg, float('inf'))
                                if current_priority < hit_group_priority:
                                    hit_group = hg
                                    hit_group_priority = current_priority
                        
                        has_damage = total_damage > 0
                        
                        # Create a DataPoint object instead of a tuple
                        data_point = DataPoint(
                            map_name=map_name,
                            attacker_x=attacker_x,
                            attacker_y=attacker_y,
                            attacker_z=attacker_z,
                            attacker_view_x=attacker_view_x,
                            victim_x=victim_x,
                            victim_y=victim_y,
                            victim_z=victim_z,
                            victim_view_x=victim_view_x,
                            attacker_hp=attacker_hp,
                            attacker_weapon=attacker_weapon,
                            victim_has_helmet=victim_has_helmet,
                            victim_has_armor=victim_has_armor,
                            has_damage=has_damage,
                            total_damage=total_damage,
                            hit_group=hit_group
                        )
                        local_data.append(data_point)
        
        return local_data

    def get_hit_group_class_weights(self):
        """
        Compute class weights for hit groups based on inverse frequency.
        Only considers positive samples (where damage occurred).
        
        Returns:
            torch.Tensor: Tensor of class weights for each hit group
        """
        # Count occurrences of each hit group in positive samples
        hit_group_counts = {group: 0 for group in self.hit_group_names}
        
        # Only consider positive samples (where damage occurred)
        for data_point in self.positive_samples:
            dp = DataPoint.from_tuple(data_point)
            if dp.hit_group is not None:
                hit_group_counts[dp.hit_group] += 1
        
        # Calculate inverse frequency
        total_samples = sum(hit_group_counts.values())
        class_weights = []
        
        for group in self.hit_group_names:
            count = hit_group_counts[group]
            # Handle case where a class might not be present
            if count == 0:
                weight = 1.0  # Default weight for missing classes
            else:
                weight = total_samples / (len(self.hit_group_names) * count)
            class_weights.append(weight)
        
        return torch.tensor(class_weights, dtype=torch.float32)


def collate_fn(batch):
    """Collate function for DataLoader that separates features (X) and targets (Y).
    
    Args:
        batch: List of dictionaries, each containing:
            'map': One-hot encoded map tensor
            'coords_and_angles': Normalized position data tensor
            'weapon': One-hot encoded weapon tensor
            'armor_features': Boolean features tensor
            'targets': Target values tensor
            'hit_group': One-hot encoded hit group tensor
    
    Returns:
        tuple: (features_dict, targets) where:
            features_dict: Dictionary containing batched X features
            targets: Dictionary containing batched Y targets and hit group
    """
    # Separate X and Y
    maps = torch.stack([item['map'] for item in batch])
    coords_and_angles = torch.stack([item['coords_and_angles'] for item in batch])
    weapons = torch.stack([item['weapon'] for item in batch])
    armor_features = torch.stack([item['armor_features'] for item in batch])
    damage_indicator = torch.stack([item['damage_indicator'] for item in batch])
    damage_value = torch.stack([item['damage_value'] for item in batch])
    hit_group = torch.stack([item['hit_group'] for item in batch])
    
    # Group all X features in a dictionary
    features_dict = {
        'map': maps,
        'coords_and_angles': coords_and_angles,
        'weapon': weapons,
        'armor_features': armor_features
    }
    
    # Include hit_group in targets or as a separate output
    targets_dict = {
        'damage_indicator': damage_indicator,
        'damage_value': damage_value,
        'hit_group': hit_group
    }
    
    return features_dict, targets_dict


if __name__ == "__main__":
    data_dir = os.path.join("..", "data", "player_seq_allmap_full_npz_damage")
    file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')][:4]

    positive_samples_file = os.path.join("..", "data", "positive_samples.npy")
    positive_samples = np.load(positive_samples_file)[:10000]
    
    dataset = DamageOutcomeDataset(file_paths, positive_samples)
    dataset.reload_data()
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=collate_fn)
    for batch in dataloader:
        print(batch)
        break

    print(dataset.current_file_idx)
    dataset.reload_data()
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=collate_fn)
    for batch in dataloader:
        print(batch)
        break
    print(dataset.current_file_idx)

    dataset.reload_data()
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=collate_fn)
    for batch in dataloader:
        print(batch)
        break
    print(dataset.current_file_idx)