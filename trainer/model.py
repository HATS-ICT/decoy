import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DamageOutcomeModel(nn.Module):
    def __init__(self, num_maps, num_weapons, num_hit_groups, 
                 hidden_dim=128, dropout=0.3, use_xavier_init=False, use_batchnorm=False, mode="joint"):
        """
        Args:
            num_maps: Number of map features
            num_weapons: Number of weapon features
            num_hit_groups: Number of hit group categories
            hidden_dim: Hidden dimension size
            use_xavier_init: Whether to use Xavier (Glorot) initialization
            use_batchnorm: Whether to use batch normalization
            mode: Mode to use for the model ("joint", "causal", "causal_embed")
        """
        super(DamageOutcomeModel, self).__init__()
        
        self.num_hit_groups = num_hit_groups
        self.mode = mode
        
        # Create hit group embeddings for causal_embed mode
        if mode == "causal_embed":
            self.hit_group_embedding = nn.Embedding(num_hit_groups, hidden_dim // 4)
        
        self.map_embed = nn.Sequential(
            nn.Linear(num_maps, hidden_dim // 4),
            nn.ReLU()
        )
        
        self.coords_embed = nn.Sequential(
            nn.Linear(11, hidden_dim // 2),  # 9 coordinate and angle features + 2 engineered features
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU()
        )
        
        self.weapon_embed = nn.Sequential(
            nn.Linear(num_weapons, hidden_dim // 4),
            nn.ReLU()
        )
        
        self.armor_embed = nn.Sequential(
            nn.Linear(2, hidden_dim // 8),
            nn.ReLU()
        )
        
        combined_input_dim = hidden_dim // 4 + hidden_dim // 2 + hidden_dim // 4 + hidden_dim // 8
        
        self.combined_layers = nn.Sequential(
            nn.Linear(combined_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.damage_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Define damage regressor with appropriate input dimensions based on mode
        if mode == "joint":
            damage_input_dim = hidden_dim
        elif mode == "causal":
            damage_input_dim = hidden_dim + num_hit_groups
        elif mode == "causal_embed":
            damage_input_dim = hidden_dim + hidden_dim // 4
        
        self.damage_regressor = nn.Sequential(
            nn.Linear(damage_input_dim, 1),
            # nn.ReLU()  # Damage can't be negative
        )
        self.hit_group_classifier = nn.Linear(hidden_dim, num_hit_groups)
        
        if use_xavier_init:
            self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier (Glorot) initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        map_features = self.map_embed(x['map'])
        coords_features = self.coords_embed(x['coords_and_angles'])
        weapon_features = self.weapon_embed(x['weapon'])
        armor_features = self.armor_embed(x['armor_features'])
        
        combined = torch.cat([
            map_features, 
            coords_features, 
            weapon_features, 
            armor_features
        ], dim=1)
        
        latent_features = self.combined_layers(combined)

        damage_indicator_probs = self.damage_classifier(latent_features)
        hit_group_logits = self.hit_group_classifier(latent_features)
        
        # Different damage prediction based on mode
        if self.mode == "joint":
            # Original approach - direct prediction
            damage_value = self.damage_regressor(latent_features)
        
        elif self.mode == "causal":
            # Use hit group probabilities as input for damage prediction
            hit_group_probs = F.softmax(hit_group_logits, dim=1)
            damage_features = torch.cat([latent_features, hit_group_probs], dim=1)
            damage_value = self.damage_regressor(damage_features)
        
        elif self.mode == "causal_embed":
            # Use learned hit group embeddings weighted by probabilities
            hit_group_probs = F.softmax(hit_group_logits, dim=1).detach()
            
            # Create weighted embeddings using matrix multiplication
            # First get all embeddings as a matrix [num_hit_groups, hidden_dim//4]
            all_embeddings = self.hit_group_embedding.weight
            
            # Matrix multiply: [batch_size, num_hit_groups] @ [num_hit_groups, hidden_dim//4]
            hit_group_context = torch.matmul(hit_group_probs, all_embeddings)
            
            damage_features = torch.cat([latent_features, hit_group_context], dim=1)
            damage_value = self.damage_regressor(damage_features)
        
        result = {
            'damage_indicator_probs': damage_indicator_probs,
            'damage_value': damage_value,
            'hit_group_logits': hit_group_logits
        }
        return result