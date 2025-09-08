import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GameStateEncoder(nn.Module):
    def __init__(self, num_maps, num_weapons, hidden_dim=128, use_xavier_init=False, use_batchnorm=False):
        """
        Encodes game state features (map, coordinates, weapon, armor)
        
        Args:
            num_maps: Number of map features
            num_weapons: Number of weapon features
            hidden_dim: Hidden dimension size
            use_xavier_init: Whether to use Xavier (Glorot) initialization
            use_batchnorm: Whether to use batch normalization
        """
        super(GameStateEncoder, self).__init__()
        
        feature_hidden_dim = hidden_dim // 8

        # Feature embedding layers
        self.map_embed = nn.Sequential(
            nn.Linear(num_maps, feature_hidden_dim),
            nn.BatchNorm1d(feature_hidden_dim) if use_batchnorm else nn.Identity(),
            nn.ReLU()
        )
        
        self.coords_embed = nn.Sequential(
            nn.Linear(11, feature_hidden_dim),  # 9 coordinate and angle features + 2 engineered features
            nn.BatchNorm1d(feature_hidden_dim) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
        )
        
        self.weapon_embed = nn.Sequential(
            nn.Linear(num_weapons, feature_hidden_dim),
            nn.BatchNorm1d(feature_hidden_dim) if use_batchnorm else nn.Identity(),
            nn.ReLU()
        )
        
        self.armor_embed = nn.Sequential(
            nn.Linear(2, feature_hidden_dim),
            nn.BatchNorm1d(feature_hidden_dim) if use_batchnorm else nn.Identity(),
            nn.ReLU()
        )
        
        # Combined feature dimension
        self.output_dim = feature_hidden_dim * 4
        
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
        """Extract and combine conditional feature embeddings"""
        map_features = self.map_embed(x['map'])
        coords_features = self.coords_embed(x['coords_and_angles'])
        weapon_features = self.weapon_embed(x['weapon'])
        armor_features = self.armor_embed(x['armor_features'])
        
        # Combine all conditional features
        conditional_features = torch.cat([
            map_features, 
            coords_features, 
            weapon_features, 
            armor_features
        ], dim=1)
        
        return conditional_features


class DamageOutcomeVAEEncoder(nn.Module):
    def __init__(self, conditional_features_dim, num_hit_groups, hidden_dim=128, 
                 latent_dim=16, dropout=0.3, use_xavier_init=False, use_batchnorm=False):
        """
        VAE Encoder for damage outcome
        
        Args:
            conditional_features_dim: Dimension of conditional features
            num_hit_groups: Number of hit group categories
            hidden_dim: Hidden dimension size
            latent_dim: Dimension of the latent space
            dropout: Dropout rate
            use_xavier_init: Whether to use Xavier (Glorot) initialization
            use_batchnorm: Whether to use batch normalization
        """
        super(DamageOutcomeVAEEncoder, self).__init__()

        feature_hidden_dim = hidden_dim // 8

        # Feature embedding layers
        self.damage_value_embed = nn.Sequential(
            nn.Linear(1, feature_hidden_dim),
            nn.BatchNorm1d(feature_hidden_dim) if use_batchnorm else nn.Identity(),
            nn.ReLU()
        )

        self.hit_group_embed = nn.Sequential(
            nn.Linear(num_hit_groups, feature_hidden_dim),  
            nn.BatchNorm1d(feature_hidden_dim) if use_batchnorm else nn.Identity(),
            nn.ReLU()
        )

        self.encoder_input_dim = feature_hidden_dim * 2 + conditional_features_dim
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(self.encoder_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # VAE components
        self.mu_layer = nn.Linear(hidden_dim // 4, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim // 4, latent_dim)
        
        if use_xavier_init:
            self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier (Glorot) initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, damage_value, hit_group_onehot, conditional_features):
        """
        Encode damage value and hit group to latent space parameters (mu, logvar)
        
        Args:
            damage_value: Target damage value tensor
            hit_group_onehot: Hit group one-hot tensor
            conditional_features: Conditional features from GameStateEncoder
        """
        damage_value_features = self.damage_value_embed(damage_value)
        hit_group_features = self.hit_group_embed(hit_group_onehot) 
        # Concatenate damage value, hit group and conditional features
        encoder_input = torch.cat([damage_value_features, hit_group_features, conditional_features], dim=1)
        
        # Process combined input
        encoded = self.encoder(encoder_input)
        mu = self.mu_layer(encoded)
        logvar = self.logvar_layer(encoded)
        
        return mu, logvar


class DamageOutcomeVAEDecoder(nn.Module):
    def __init__(self, conditional_features_dim, num_hit_groups, hidden_dim=128, 
                 latent_dim=16, dropout=0.3, use_xavier_init=False, use_batchnorm=False):
        """
        VAE Decoder for damage outcome
        
        Args:
            conditional_features_dim: Dimension of conditional features
            num_hit_groups: Number of hit group categories
            hidden_dim: Hidden dimension size
            latent_dim: Dimension of the latent space
            dropout: Dropout rate
            use_xavier_init: Whether to use Xavier (Glorot) initialization
            use_batchnorm: Whether to use batch normalization
        """
        super(DamageOutcomeVAEDecoder, self).__init__()
        
        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + conditional_features_dim, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Damage regressor
        self.damage_regressor = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )
        
        # Hit group classifier
        self.hit_group_classifier = nn.Linear(hidden_dim, num_hit_groups)
        
        self.num_hit_groups = num_hit_groups
        
        if use_xavier_init:
            self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier (Glorot) initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, z, conditional_features):
        """
        Decode from latent space to damage prediction and hit group prediction
        
        Args:
            z: Latent vector
            conditional_features: Conditional features from GameStateEncoder
        """
        # Concatenate latent vector with conditional inputs
        decoder_input = torch.cat([z, conditional_features], dim=1)
        decoded = self.decoder(decoder_input)
        
        # Predict damage value
        damage_value = self.damage_regressor(decoded)
        
        # Predict hit group
        hit_group_logits = self.hit_group_classifier(decoded)
        
        return damage_value, hit_group_logits


class DamageOutcomeGenerator(nn.Module):
    def __init__(self, num_maps, num_weapons, num_hit_groups, 
                 hidden_dim=128, latent_dim=16, dropout=0.3, 
                 use_xavier_init=False, use_batchnorm=False):
        """
        Conditional Variational Autoencoder for damage regression and hit group prediction
        
        Args:
            num_maps: Number of map features
            num_weapons: Number of weapon features
            num_hit_groups: Number of hit group categories
            hidden_dim: Hidden dimension size
            latent_dim: Dimension of the latent space
            dropout: Dropout rate
            use_xavier_init: Whether to use Xavier (Glorot) initialization
            use_batchnorm: Whether to use batch normalization
        """
        super(DamageOutcomeGenerator, self).__init__()
        
        # Create the three component modules
        self.game_state_encoder = GameStateEncoder(
            num_maps=num_maps,
            num_weapons=num_weapons,
            hidden_dim=hidden_dim,
            use_xavier_init=use_xavier_init,
            use_batchnorm=use_batchnorm
        )
        
        self.vae_encoder = DamageOutcomeVAEEncoder(
            conditional_features_dim=self.game_state_encoder.output_dim,
            num_hit_groups=num_hit_groups,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
            use_xavier_init=use_xavier_init,
            use_batchnorm=use_batchnorm
        )
        
        self.vae_decoder = DamageOutcomeVAEDecoder(
            conditional_features_dim=self.game_state_encoder.output_dim,
            num_hit_groups=num_hit_groups,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
            use_xavier_init=use_xavier_init,
            use_batchnorm=use_batchnorm
        )
        
        self.latent_dim = latent_dim
        self.num_hit_groups = num_hit_groups
    
    def encode(self, x, damage_value, hit_group_onehot):
        """
        Encode damage value and hit group to latent space parameters (mu, logvar)
        
        Args:
            x: Dictionary of input features
            damage_value: Target damage value tensor
            hit_group_onehot: Hit group one-hot tensor
        """
        conditional_features = self.game_state_encoder(x)
        mu, logvar = self.vae_encoder(damage_value, hit_group_onehot, conditional_features)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick: sample from latent space"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z, x):
        """
        Decode from latent space to damage prediction and hit group prediction
        
        Args:
            z: Latent vector
            x: Dictionary of input features
        """
        conditional_features = self.game_state_encoder(x)
        damage_value, hit_group_logits = self.vae_decoder(z, conditional_features)
        return damage_value, hit_group_logits
    
    def forward(self, x, damage_value=None, hit_group_onehot=None):
        """
        Forward pass through the VAE
        
        Args:
            x: Dictionary of input features
            damage_value: Target damage value tensor (required for training, optional for inference)
            hit_group_onehot: Target hit group one-hot tensor (required for training, optional for inference)
        """
        
        # If damage_value and hit_group_onehot are provided (training mode)
        if damage_value is not None and hit_group_onehot is not None:
            mu, logvar = self.encode(x, damage_value, hit_group_onehot) 
            z = self.reparameterize(mu, logvar)
            reconstructed_damage, hit_group_logits = self.decode(z, x)
            return reconstructed_damage, hit_group_logits, mu, logvar
        
        # If no damage_value and hit_group_onehot (inference mode)
        else:
            # Sample from prior N(0, 1)
            batch_size = x['map'].size(0)
            z = torch.randn(batch_size, self.latent_dim, device=x['map'].device)
            damage_value, hit_group_logits = self.decode(z, x)
            return damage_value, hit_group_logits, None, None
    
    def sample(self, x, num_samples=1, use_mean=False):
        """
        Sample multiple predictions for the same input
        
        Args:
            x: Dictionary of input features
            num_samples: Number of samples to generate
            use_mean: Whether to use the mean of the prior (instead of sampling)
        """
        # Get conditional features
        conditional_features = self.game_state_encoder(x)
        
        # Expand dimensions for multiple samples
        batch_size = conditional_features.size(0)
        conditional_expanded = conditional_features.unsqueeze(1).expand(
            batch_size, num_samples, self.game_state_encoder.output_dim)
        conditional_reshaped = conditional_expanded.reshape(
            batch_size * num_samples, self.game_state_encoder.output_dim)
        
        # Sample from prior N(0, 1)
        if use_mean:
            z = torch.zeros(batch_size * num_samples, self.latent_dim, device=x['map'].device)
        else:
            z = torch.randn(batch_size * num_samples, self.latent_dim, device=x['map'].device)
        
        # Decode
        x_expanded = {
            'map': x['map'].unsqueeze(1).expand(batch_size, num_samples, -1).reshape(batch_size * num_samples, -1),
            'coords_and_angles': x['coords_and_angles'].unsqueeze(1).expand(batch_size, num_samples, -1).reshape(batch_size * num_samples, -1),
            'weapon': x['weapon'].unsqueeze(1).expand(batch_size, num_samples, -1).reshape(batch_size * num_samples, -1),
            'armor_features': x['armor_features'].unsqueeze(1).expand(batch_size, num_samples, -1).reshape(batch_size * num_samples, -1)
        }
        
        damage_values, hit_group_logits = self.decode(z, x_expanded)
        
        # Reshape back to [batch_size, num_samples, 1] for damage
        # and [batch_size, num_samples, num_hit_groups] for hit group
        return (
            damage_values.reshape(batch_size, num_samples, 1),
            hit_group_logits.reshape(batch_size, num_samples, self.num_hit_groups)
        )


class DamageIndicatorPredictor(nn.Module):
    def __init__(self, num_maps, num_weapons, hidden_dim=128, dropout=0.3, 
                 use_xavier_init=False, use_batchnorm=False):
        """
        Binary classifier to predict whether damage will occur
        
        Args:
            num_maps: Number of map features
            num_weapons: Number of weapon features
            hidden_dim: Hidden dimension size
            dropout: Dropout rate
            use_xavier_init: Whether to use Xavier (Glorot) initialization
            use_batchnorm: Whether to use batch normalization
        """
        super(DamageIndicatorPredictor, self).__init__()
        
        # Reuse the same game state encoder
        self.game_state_encoder = GameStateEncoder(
            num_maps=num_maps,
            num_weapons=num_weapons,
            hidden_dim=hidden_dim,
            use_xavier_init=use_xavier_init
        )
        
        # MLP layers for binary classification
        self.classifier = nn.Sequential(
            nn.Linear(self.game_state_encoder.output_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
            # No activation here - will use sigmoid in loss function
        )
        
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
        """
        Forward pass through the binary classifier
        
        Args:
            x: Dictionary of input features
        
        Returns:
            Logit for binary classification (apply sigmoid for probability)
        """
        # Get conditional features from game state encoder
        conditional_features = self.game_state_encoder(x)
        
        # Pass through classifier
        logits = self.classifier(conditional_features)
        
        return logits