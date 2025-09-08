import os
import json
from typing import Dict, Tuple, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .config import DAMAGE_INDICATOR_PREDICTOR_MODEL_PATH, DAMAGE_OUTCOME_GENERATOR_MODEL_PATH
from .damage_estimate_model import DamageOutcomeGenerator, DamageIndicatorPredictor
from .utils import transform_panda3d_to_csgo


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

class DamageEstimateModel:
    """
    Wrapper class for damage prediction models:
    1. DamageIndicatorPredictor - predicts if damage will occur
    2. DamageOutcomeGenerator - generates damage amount and hit group if damage occurs
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_weapons = len(WEAPON_NAMES)
        self.num_maps = len(MAP_NAMES)  # Keep the full map count for model loading
        self.num_hit_groups = len(HIT_GROUP_NAMES)  # Use the correct hit group count
        
        # Default map and weapon indices
        self.default_map_index = MAP_NAMES.index('de_dust2')
        self.default_weapon_index = WEAPON_NAMES.index('AK-47')
        
        # Models will be loaded later
        self.indicator_model = None
        self.generator_model = None
        
        # Hit group names for reference
        self.hit_group_names = HIT_GROUP_NAMES
        
        # Default thresholds
        self.damage_indicator_threshold = 0.5
        self.load_model()
        
    def load_model(self) -> None:
        """
        Load both damage indicator and generator models from the specified path
        
        Args:
            model_path: Base directory containing both model folders
        """
        # Find the indicator and generator model folders
        indicator_folder = DAMAGE_INDICATOR_PREDICTOR_MODEL_PATH
        generator_folder = DAMAGE_OUTCOME_GENERATOR_MODEL_PATH
        
        # print(f"Loading damage indicator model from {indicator_folder}")
        # print(f"Loading damage generator model from {generator_folder}")
        
        self._load_indicator_model(indicator_folder)
        self._load_generator_model(generator_folder)
    
    def _load_indicator_model(self, model_folder: str) -> None:
        """Load the damage indicator model"""
        # Find the best model checkpoint
        checkpoint_path = os.path.join(model_folder, "checkpoints", "best_model.pth")
        args_path = os.path.join(model_folder, "args.json")
        with open(args_path, 'r') as f:
            args = json.load(f)
        
        # Create the model
        self.indicator_model = DamageIndicatorPredictor(
            num_maps=len(MAP_NAMES),
            num_weapons=len(WEAPON_NAMES),
            hidden_dim=args.get('hidden_dim', 256),
            dropout=args.get('dropout', 0.3),
            use_xavier_init=False,  # No need for initialization when loading weights
            use_batchnorm=args.get('use_batchnorm', True)
        ).to(self.device)
        
        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.indicator_model.load_state_dict(checkpoint['model_state_dict'])
        self.indicator_model.eval()
        
        # Get optimal threshold from results if available
        try:
            metrics_path = os.path.join(model_folder.replace("logs", "results"), "metrics", "test_metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                    self.damage_indicator_threshold = metrics.get('optimal_threshold', 0.5)
                    print(f"Using optimal damage indicator threshold: {self.damage_indicator_threshold}")
        except Exception as e:
            print(f"Could not load optimal threshold: {e}")
            print("Using default threshold of 0.5")
    
    def _load_generator_model(self, model_folder: str) -> None:
        """Load the damage outcome generator model"""
        # Find the best model checkpoint
        checkpoint_path = os.path.join(model_folder, "checkpoints", "best_model.pth")
        args_path = os.path.join(model_folder, "args.json")
        with open(args_path, 'r') as f:
            args = json.load(f)
        
        # Create the model with correct dimensions
        self.generator_model = DamageOutcomeGenerator(
            num_maps=len(MAP_NAMES),
            num_weapons=len(WEAPON_NAMES),
            num_hit_groups=len(HIT_GROUP_NAMES),
            hidden_dim=args.get('hidden_dim', 256),
            latent_dim=args.get('latent_dim', 8),
            dropout=args.get('dropout', 0.3),
            use_xavier_init=False,  # No need for initialization when loading weights
            use_batchnorm=args.get('use_batchnorm', True)
        ).to(self.device)
        
        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.generator_model.load_state_dict(checkpoint['model_state_dict'])
        self.generator_model.eval()
    
    def predict_damage(self, 
                       attacker_pos: np.ndarray,
                       victim_pos: np.ndarray, 
                       attacker_angle: np.ndarray,
                       victim_angle: np.ndarray,
                       attacker_hp: int,
                       attacker_weapon_id: int,
                       victim_has_armor: bool = False,
                       victim_has_helmet: bool = False) -> Tuple[bool, Optional[float], Optional[int]]:
        """
        Predict if damage will occur and the damage amount if it does
        
        Args:
            attacker_pos: Position of the attacker (x, y, z)
            victim_pos: Position of the victim (x, y, z)
            attacker_angle: Horizontal view angle of the attacker (yaw)
            victim_angle: Horizontal view angle of the victim (yaw)
            attacker_hp: Health of the attacker
            attacker_weapon_id: ID of the weapon being used
            victim_has_armor: Whether the victim has armor
            victim_has_helmet: Whether the victim has a helmet
            
        Returns:
            Tuple of (will_damage, damage_amount, hit_group)
            - will_damage: Boolean indicating if damage will occur
            - damage_amount: Predicted damage amount (None if no damage)
            - hit_group: Predicted hit group (None if no damage)
        """
        if self.indicator_model is None or self.generator_model is None:
            raise ValueError("Models not loaded. Call load_model() first.")
        
        # Prepare input features
        features = self._prepare_features(
            attacker_pos, victim_pos, attacker_angle, victim_angle, attacker_hp, attacker_weapon_id, victim_has_armor, victim_has_helmet
        )

        # Predict if damage will occur
        with torch.no_grad():
            damage_indicator_logits = self.indicator_model(features)
            damage_indicator_prob = torch.sigmoid(damage_indicator_logits).item()
            will_damage = damage_indicator_prob >= self.damage_indicator_threshold
        
        # If no damage predicted, return early
        if not will_damage:
            return False, None, None
        
        # If damage predicted, generate damage outcome
        with torch.no_grad():
            # Sample multiple outcomes and take the mean
            damage_values, hit_group_logits = self.generator_model.sample(
                features, num_samples=1
            )
            
            # Average damage values across samples
            damage_amount = damage_values.mean().item() * 100
            
            # Get most likely hit group across samples
            hit_group_probs = F.softmax(hit_group_logits, dim=2)
            hit_group_probs_mean = hit_group_probs.mean(dim=1)
            hit_group = hit_group_probs_mean.argmax(dim=1).item()
        
        return True, damage_amount, hit_group
    
    def _prepare_features(self, 
                          attacker_pos: np.ndarray,
                          victim_pos: np.ndarray, 
                          attacker_angle: np.ndarray,
                          victim_angle: np.ndarray,
                          attacker_hp: int,
                          attacker_weapon_id: int,
                          victim_has_armor: bool,
                          victim_has_helmet: bool) -> Dict[str, torch.Tensor]:
        """
        Prepare input features for the models
        
        Args:
            attacker_pos: Position of the attacker (x, y, z)
            victim_pos: Position of the victim (x, y, z)
            attacker_angle: Horizontal view angle of the attacker (yaw)
            victim_angle: Horizontal view angle of the victim (yaw)
            attacker_hp: Health of the attacker
            attacker_weapon_id: ID of the weapon being used
            victim_has_armor: Whether the victim has armor
            victim_has_helmet: Whether the victim has a helmet
            
        Returns:
            Dictionary of input features as tensors
        """
        # Extract position data
        attacker_x, attacker_y, attacker_z = transform_panda3d_to_csgo(attacker_pos)
        victim_x, victim_y, victim_z = transform_panda3d_to_csgo(victim_pos)
        
        # Calculate distance between players
        distance = torch.sqrt(
            torch.tensor((attacker_x - victim_x)**2 + 
                         (attacker_y - victim_y)**2 + 
                         (attacker_z - victim_z)**2)
        )
        
        # Calculate relative angle
        dx = victim_x - attacker_x
        dy = victim_y - attacker_y
        
        attacker_view_rad = torch.tensor(attacker_angle * np.pi / 180.0)
        vector_angle = torch.atan2(torch.tensor(dy), torch.tensor(dx))
        relative_angle = torch.remainder(vector_angle - attacker_view_rad + np.pi, 2 * np.pi) - np.pi
        normalized_relative_angle = relative_angle / np.pi
        
        # Combine position and angle features
        coords_and_angles = torch.tensor([
            attacker_x, attacker_y, attacker_z,  # attacker xyz
            float(attacker_angle) / 360.0,  # normalized attacker_view_x
            victim_x, victim_y, victim_z,  # victim xyz
            float(victim_angle) / 360.0,  # normalized victim_view_x
            float(attacker_hp) / 100.0,  # normalized attacker_hp
            distance.item(),  # distance between attacker and victim
            normalized_relative_angle.item(),  # relative angle between attacker's view and victim
        ], dtype=torch.float32)
        
        # One-hot encode weapon
        weapon_onehot = np.zeros(self.num_weapons, dtype=np.float32)
        weapon_onehot[attacker_weapon_id] = 1.0
        
        # Armor features
        armor_features = torch.tensor([
            float(victim_has_helmet),  # victim_has_helmet
            float(victim_has_armor),  # victim_has_armor
        ], dtype=torch.float32)
        
        # Map features (always use de_dust2)
        map_features = np.zeros(self.num_maps, dtype=np.float32)
        map_features[self.default_map_index] = 1.0
        
        # Convert to tensors and add batch dimension
        features = {
            'map': torch.tensor(map_features, dtype=torch.float32).unsqueeze(0).to(self.device),
            'coords_and_angles': coords_and_angles.unsqueeze(0).to(self.device),
            'weapon': torch.tensor(weapon_onehot, dtype=torch.float32).unsqueeze(0).to(self.device),
            'armor_features': armor_features.unsqueeze(0).to(self.device)
        }
        
        return features 
    
    