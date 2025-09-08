import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataset import DamageOutcomeDataset, collate_fn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import os
import argparse
from utils import create_run_dir, calculate_regression_metrics, plot_regression_scatter, plot_damage_distribution_matrix
import matplotlib.pyplot as plt
import wandb


class DamageRegressionVAE(nn.Module):
    def __init__(self, num_maps, num_weapons, num_hit_groups, 
                 hidden_dim=128, latent_dim=16, dropout=0.3, 
                 use_xavier_init=False, use_batchnorm=False):
        """
        Conditional Variational Autoencoder for damage regression
        
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
        super(DamageRegressionVAE, self).__init__()
        
        # Feature embedding layers (same as in regression model)
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
        
        self.hit_group_embed = nn.Sequential(
            nn.Linear(num_hit_groups, hidden_dim // 4),
            nn.ReLU()
        )
        
        # Combined feature dimension
        self.combined_input_dim = hidden_dim // 4 + hidden_dim // 2 + hidden_dim // 4 + hidden_dim // 8 + hidden_dim // 4
        
        # Encoder network - takes only damage value as input
        self.encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),  # Only damage value input
            nn.BatchNorm1d(hidden_dim) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # VAE components
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder network - takes latent vector and conditional features
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + self.combined_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Damage regressor
        self.damage_regressor = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.ReLU()
        )
        
        self.latent_dim = latent_dim
        
        if use_xavier_init:
            self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier (Glorot) initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def encode(self, damage_value):
        """
        Encode damage value to latent space parameters (mu, logvar)
        
        Args:
            damage_value: Target damage value tensor
        """
        # Process damage value only
        encoded = self.encoder(damage_value)
        mu = self.mu_layer(encoded)
        logvar = self.logvar_layer(encoded)
        
        return mu, logvar
    
    def get_conditional_features(self, x):
        """Extract and combine conditional feature embeddings"""
        map_features = self.map_embed(x['map'])
        coords_features = self.coords_embed(x['coords_and_angles'])
        weapon_features = self.weapon_embed(x['weapon'])
        armor_features = self.armor_embed(x['armor_features'])
        hit_group_features = self.hit_group_embed(x['hit_group_onehot'])
        
        # Combine all conditional features
        conditional_features = torch.cat([
            map_features, 
            coords_features, 
            weapon_features, 
            armor_features,
            hit_group_features
        ], dim=1)
        
        return conditional_features
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick: sample from latent space"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z, conditional_features):
        """
        Decode from latent space to damage prediction
        
        Args:
            z: Latent vector
            conditional_features: Conditional input features
        """
        # Concatenate latent vector with conditional inputs
        decoder_input = torch.cat([z, conditional_features], dim=1)
        decoded = self.decoder(decoder_input)
        damage_value = self.damage_regressor(decoded)
        damage_value = torch.clamp(damage_value, min=0, max=1)
        return damage_value
    
    def forward(self, x, damage_value=None):
        """
        Forward pass through the VAE
        
        Args:
            x: Dictionary of input features
            damage_value: Target damage value tensor (required for training, optional for inference)
        """
        # Get conditional features
        conditional_features = self.get_conditional_features(x)
        
        # If damage_value is provided (training mode)
        if damage_value is not None:
            mu, logvar = self.encode(damage_value)
            z = self.reparameterize(mu, logvar)
            reconstructed_damage = self.decode(z, conditional_features)
            return reconstructed_damage, mu, logvar
        
        # If no damage_value (inference mode)
        else:
            # Sample from prior N(0, 1)
            batch_size = x['map'].size(0)
            z = torch.randn(batch_size, self.latent_dim, device=x['map'].device)
            damage_value = self.decode(z, conditional_features)
            return damage_value, None, None
    
    def sample(self, x, num_samples=1, use_mean=False):
        """
        Sample multiple predictions for the same input
        
        Args:
            x: Dictionary of input features
            num_samples: Number of samples to generate
        """
        # Get conditional features
        conditional_features = self.get_conditional_features(x)
        
        # Expand dimensions for multiple samples
        batch_size = conditional_features.size(0)
        conditional_expanded = conditional_features.unsqueeze(1).expand(batch_size, num_samples, self.combined_input_dim)
        conditional_reshaped = conditional_expanded.reshape(batch_size * num_samples, self.combined_input_dim)
        
        # Sample from prior N(0, 1)
        if use_mean:
            z = torch.zeros(batch_size * num_samples, self.latent_dim, device=x['map'].device)
        else:
            z = torch.randn(batch_size * num_samples, self.latent_dim, device=x['map'].device)
        
        # Decode
        damage_values = self.decode(z, conditional_reshaped)
        
        # Reshape back to [batch_size, num_samples, 1]
        return damage_values.reshape(batch_size, num_samples, 1)


class VAERegressionTrainer:
    def __init__(self, model, run_dir, train_dataset, val_dataset, device, args):
        self.model = model
        self.run_dir = run_dir
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.args = args
        self.optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.kl_weight = args.kl_weight  # Weight for KL divergence term

    def train(self, verbose: bool = True):
        for epoch in range(self.args.num_epoch):
            self.current_epoch = epoch
            train_loss, train_metrics = self.train_epoch()
            val_loss, val_metrics = self.validate_epoch()
            
            # Save metrics
            self._log_metrics(train_metrics, val_metrics)
            
            # Save model if it's the best so far
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best_model.pth")
            
            print(f"Epoch {epoch+1}/{self.args.num_epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Train MAE: {train_metrics['mae']:.2f}, Val MAE: {val_metrics['mae']:.2f}")
            print(f"Train R²: {train_metrics['r2_score']:.4f}, Val R²: {val_metrics['r2_score']:.4f}")
            print(f"Train KL Loss: {train_metrics['kl_loss']:.4f}, Val KL Loss: {val_metrics['kl_loss']:.4f}")
            print(f"Train Recon Loss: {train_metrics['recon_loss']:.4f}, Val Recon Loss: {val_metrics['recon_loss']:.4f}")
            print(f"Train Total Loss: {train_metrics['loss']:.4f}, Val Total Loss: {val_metrics['loss']:.4f}")
            print(f"KL Weight: {self.kl_weight}")
            print("--------------------------------")
        if self.args.use_wandb:
            wandb.finish()
        return
    
    def _log_metrics(self, train_metrics, val_metrics):
        # Log to wandb if enabled
        if self.args.use_wandb:
            wandb.log({
                'epoch': self.current_epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_recon_loss': train_metrics['recon_loss'],
                'train_kl_loss': train_metrics['kl_loss'],
                'train_mae': train_metrics['mae'],
                'train_mse': train_metrics['mse'],
                'train_rmse': train_metrics['rmse'],
                'train_r2': train_metrics['r2_score'],
                'val_loss': val_metrics['loss'],
                'val_recon_loss': val_metrics['recon_loss'],
                'val_kl_loss': val_metrics['kl_loss'],
                'val_mae': val_metrics['mae'],
                'val_mse': val_metrics['mse'],
                'val_rmse': val_metrics['rmse'],
                'val_r2': val_metrics['r2_score'],
                'kl_weight': self.kl_weight  # Add KL weight to logging
            }, step=self.global_step)
        
        # Save plots
        self._save_plots(train_metrics, 'train')
        self._save_plots(val_metrics, 'val')
    
    def _save_plots(self, metrics, prefix):
        plots_dir = os.path.join(self.run_dir, "plots")
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        # Create regression scatter plot
        plot_path = os.path.join(plots_dir, f"{prefix}_regression_epoch_{self.current_epoch+1}.png")
        plot_regression_scatter(
            metrics['targets'],
            metrics['predictions'],
            metrics['r2_score'],
            f"{prefix.capitalize()} Damage Value Prediction (Epoch {self.current_epoch+1})",
            plot_path
        )

        # Create damage distribution matrix
        plot_path_matrix = os.path.join(plots_dir, f"{prefix}_damage_distribution_epoch_{self.current_epoch+1}.png")
        self.plot_damage_distribution_matrix(
            metrics['targets'],
            metrics['predictions'],
            metrics['labels']['weapon'],
            metrics['labels']['body_part'],
            plot_path_matrix
        )
        
        if self.args.use_wandb:
            wandb.log({
                f'{prefix}_regression_plot': wandb.Image(plot_path),
                f'{prefix}_damage_distribution': wandb.Image(plot_path_matrix)
            }, step=self.global_step)
    
    def plot_damage_distribution_matrix(self, targets, predictions, weapon_labels_indices, body_part_labels_indices, plot_path):
        plot_damage_distribution_matrix(targets, predictions, weapon_labels_indices, body_part_labels_indices, plot_path)
    
    def train_epoch(self):
        self.model.train()
        return self._run_epoch(self.train_dataset, is_training=True)

    def validate_epoch(self):
        self.model.eval()
        val_loss, val_metrics = self._run_epoch(self.val_dataset, is_training=False)
        
        # Generate uncertainty plots after validation
        if self.args.generate_uncertainty_plots:
            self.generate_uncertainty_plots()
            
        return val_loss, val_metrics
    
    def generate_uncertainty_plots(self):
        """Generate plots showing prediction uncertainty using the VAE's sampling capability."""
        # Create directory for uncertainty plots
        uncertainty_dir = os.path.join(self.run_dir, "uncertainty_plots", f"epoch_{self.current_epoch+1}")
        os.makedirs(uncertainty_dir, exist_ok=True)
        
        # Load a batch of data
        self.val_dataset.load_next_chunk()
        data_loader = DataLoader(
            self.val_dataset,
            batch_size=10,  # Small batch for visualization
            shuffle=True,
            collate_fn=self._regression_collate_fn
        )
        
        # Get a single batch
        try:
            batch_x, batch_y = next(iter(data_loader))
        except StopIteration:
            print("No data available for uncertainty plots")
            return
            
        # Skip if batch is empty
        if isinstance(batch_y, torch.Tensor) and batch_y.numel() == 0:
            print("Empty batch for uncertainty plots")
            return
        
        # Move to device
        batch_x = {k: v.to(self.device) for k, v in batch_x.items()}
        batch_y = batch_y.to(self.device)
        
        # Generate multiple predictions for each sample
        with torch.no_grad():
            samples = self.model.sample(batch_x, num_samples=self.args.num_samples)
        
        # Convert to numpy for plotting
        samples_np = samples.cpu().numpy()
        true_values = np.clip(batch_y.cpu().numpy(), 0, 100)
        
        # Plot uncertainty for each sample
        for i in range(min(5, len(true_values))):  # Plot up to 5 samples
            plt.figure(figsize=(10, 6))
            
            # Get weapon and hit group info for the title
            weapon_idx = torch.argmax(batch_x['weapon'][i]).item()
            weapon_name = self.val_dataset.weapons[weapon_idx] if weapon_idx < len(self.val_dataset.weapons) else "Unknown"
            
            hit_group_idx = torch.argmax(batch_x['hit_group_onehot'][i]).item()
            hit_group_name = self.val_dataset.hit_group_names[hit_group_idx] if hit_group_idx < len(self.val_dataset.hit_group_names) else "Unknown"
            
            # Plot histogram of predictions
            plt.hist(samples_np[i, :, 0] * 100, bins=20, alpha=0.7, label='Predicted Samples')
            
            # Plot true value as a vertical line
            plt.axvline(x=true_values[i].item(), color='r', linestyle='--', 
                       label=f'True Value: {true_values[i].item():.1f}')
            
            # Calculate statistics
            mean_pred = np.mean(samples_np[i, :, 0] * 100)
            std_pred = np.std(samples_np[i, :, 0] * 100)
            
            # Plot mean as a vertical line
            plt.axvline(x=mean_pred, color='g', linestyle='-', 
                       label=f'Mean Prediction: {mean_pred:.1f} ± {std_pred:.1f}')
            
            plt.title(f'Damage Prediction Uncertainty (Epoch {self.current_epoch+1})\nWeapon: {weapon_name}, Hit Group: {hit_group_name}')
            plt.xlabel('Damage Value')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save the plot
            plt.savefig(os.path.join(uncertainty_dir, f'uncertainty_sample_{i}.png'))
            plt.close()
            
            if self.args.use_wandb:
                wandb.log({
                    f'uncertainty/sample_{i}': wandb.Image(os.path.join(uncertainty_dir, f'uncertainty_sample_{i}.png'))
                }, step=self.global_step)
        
        # Create a summary plot showing uncertainty vs error
        plt.figure(figsize=(10, 6))
        
        # Calculate mean and std for each sample
        mean_preds = np.mean(samples_np[:, :, 0] * 100, axis=1)
        std_preds = np.std(samples_np[:, :, 0] * 100, axis=1)
        true_values_flat = true_values.flatten()
        
        # Calculate absolute error
        abs_errors = np.abs(mean_preds - true_values_flat)
        
        # Plot uncertainty (std) vs absolute error
        plt.scatter(std_preds, abs_errors, alpha=0.7)
        plt.xlabel('Prediction Uncertainty (Standard Deviation)')
        plt.ylabel('Absolute Error')
        plt.title(f'Relationship Between Prediction Uncertainty and Error (Epoch {self.current_epoch+1})')
        
        # Add a trend line
        if len(std_preds) > 1:  # Need at least 2 points for a trend line
            z = np.polyfit(std_preds, abs_errors, 1)
            p = np.poly1d(z)
            plt.plot(std_preds, p(std_preds), "r--", alpha=0.7, 
                     label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
            
            # Calculate correlation
            correlation = np.corrcoef(std_preds, abs_errors)[0, 1]
            plt.annotate(f'Correlation: {correlation:.2f}', 
                         xy=(0.05, 0.95), xycoords='axes fraction',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        plt.savefig(os.path.join(uncertainty_dir, 'uncertainty_vs_error.png'))
        plt.close()
        
        if self.args.use_wandb:
            wandb.log({
                f'uncertainty/epoch_{self.current_epoch+1}/uncertainty_vs_error': wandb.Image(os.path.join(uncertainty_dir, 'uncertainty_vs_error.png'))
            }, step=self.global_step)
    
    def _run_epoch(self, dataset, is_training=True):
        """Run a single epoch of training or validation"""
        mode = "Train" if is_training else "Val"
        
        # Load data
        dataset.load_next_chunk()
        
        # Create data loader
        data_loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=is_training,
            collate_fn=self._regression_collate_fn
        )
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        all_predictions = []
        all_targets = []
        all_weapon_labels = []
        all_body_part_labels = []
        
        pbar = tqdm(data_loader, desc=f"Epoch {self.current_epoch+1}/{self.args.num_epoch} [{mode}]")
        
        if not is_training:
            # Use torch.no_grad() for validation
            with torch.no_grad():
                for batch_x, batch_y in pbar:
                    loss_dict, preds, targets = self._process_batch(batch_x, batch_y, is_training)
                    total_loss += loss_dict['total_loss'].item() * len(targets)
                    total_recon_loss += loss_dict['recon_loss'].item() * len(targets)
                    total_kl_loss += loss_dict['kl_loss'].item() * len(targets)
                    # Ensure predictions are flattened to a consistent shape
                    preds_np = preds.detach().cpu().numpy().reshape(-1, 1)
                    all_predictions.extend(preds_np)
                    all_targets.extend(targets.detach().cpu().numpy())
                    all_weapon_labels.extend(np.argmax(batch_x['weapon'].detach().cpu().numpy(), axis=1))
                    all_body_part_labels.extend(np.argmax(batch_x['hit_group_onehot'].detach().cpu().numpy(), axis=1))
                    pbar.set_postfix(loss=loss_dict['total_loss'].item())
        else:
            for batch_x, batch_y in pbar:
                loss_dict, preds, targets = self._process_batch(batch_x, batch_y, is_training)
                total_loss += loss_dict['total_loss'].item() * len(targets)
                total_recon_loss += loss_dict['recon_loss'].item() * len(targets)
                total_kl_loss += loss_dict['kl_loss'].item() * len(targets)
                # Ensure predictions are flattened to a consistent shape
                preds_np = preds.detach().cpu().numpy().reshape(-1, 1)
                all_predictions.extend(preds_np)
                all_targets.extend(targets.detach().cpu().numpy())
                all_weapon_labels.extend(np.argmax(batch_x['weapon'].detach().cpu().numpy(), axis=1))
                all_body_part_labels.extend(np.argmax(batch_x['hit_group_onehot'].detach().cpu().numpy(), axis=1))
                pbar.set_postfix(loss=loss_dict['total_loss'].item())
                self.global_step += 1
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_weapon_labels = np.array(all_weapon_labels)[:, np.newaxis]
        all_body_part_labels = np.array(all_body_part_labels)[:, np.newaxis]
        
        # Calculate metrics
        metrics = calculate_regression_metrics(all_targets, all_predictions)
        metrics['loss'] = total_loss / len(all_targets)
        metrics['recon_loss'] = total_recon_loss / len(all_targets)
        metrics['kl_loss'] = total_kl_loss / len(all_targets)
        metrics["labels"] = {
            "weapon": all_weapon_labels,
            "body_part": all_body_part_labels
        }
        metrics['predictions'] = all_predictions
        metrics['targets'] = all_targets
        
        return total_loss / len(all_targets), metrics
    
    def _process_batch(self, batch_x, batch_y, is_training=True):
        """Process a single batch"""
        # Move data to device
        batch_x = {k: v.to(self.device) for k, v in batch_x.items()}
        batch_y = batch_y.to(self.device)
        
        # Apply log transformation to damage values
        clip_damage = torch.clamp(batch_y, min=0, max=100) / 100
        
        if is_training:
            self.optimizer.zero_grad()
        
        # Forward pass - now passing damage value to the model
        outputs, mu, logvar = self.model(batch_x, clip_damage)
        
        # Calculate reconstruction loss (MSE)
        recon_loss = F.mse_loss(outputs, clip_damage)
        
        # Calculate KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / batch_y.size(0)  # Normalize by batch size
        
        # Total loss
        total_loss = recon_loss + self.kl_weight * kl_loss
        
        if is_training:
            total_loss.backward()
            if self.args.use_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gradient_clipping)
            self.optimizer.step()
        
        if self.args.use_wandb and self.global_step % self.args.log_interval == 0:
            wandb.log({
                'train_batch_loss': total_loss.item(),
                'train_batch_recon_loss': recon_loss.item(),
                'train_batch_kl_loss': kl_loss.item(),
                'kl_weight': self.kl_weight  # Add KL weight to batch logging
            }, step=self.global_step)

        if not is_training:
            # Sample from the model but ensure consistent output shape
            sampled_outputs = self.model.sample(batch_x, num_samples=1, use_mean=False)
            # Reshape to ensure consistent dimensions (batch_size, 1)
            outputs = sampled_outputs.view(-1, 1)
        
        loss_dict = {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }
        
        return loss_dict, outputs * 100, clip_damage * 100
    
    def _regression_collate_fn(self, batch):
        """Custom collate function for regression model that only uses samples with damage"""
        # Filter out samples without damage
        filtered_batch = [item for item in batch if item['damage_indicator'].item() > 0]
        
        if not filtered_batch:
            # If no samples with damage, return empty tensors
            return {
                'map': torch.empty((0, len(self.train_dataset.map_names))),
                'coords_and_angles': torch.empty((0, 11)),
                'weapon': torch.empty((0, len(self.train_dataset.weapons))),
                'armor_features': torch.empty((0, 2)),
                'hit_group_onehot': torch.empty((0, len(self.train_dataset.hit_group_names)))
            }, torch.empty((0, 1))
        
        # Stack features
        maps = torch.stack([item['map'] for item in filtered_batch])
        coords_and_angles = torch.stack([item['coords_and_angles'] for item in filtered_batch])
        weapons = torch.stack([item['weapon'] for item in filtered_batch])
        armor_features = torch.stack([item['armor_features'] for item in filtered_batch])
        
        # Convert hit_group indices to one-hot vectors
        hit_group_indices = torch.stack([item['hit_group'] for item in filtered_batch])
        hit_group_onehot = torch.zeros(len(filtered_batch), len(self.train_dataset.hit_group_names))
        hit_group_onehot.scatter_(1, hit_group_indices.unsqueeze(1), 1)
        
        damage_value = torch.stack([item['damage_value'] for item in filtered_batch])
        
        # Group all X features in a dictionary
        features_dict = {
            'map': maps,
            'coords_and_angles': coords_and_angles,
            'weapon': weapons,
            'armor_features': armor_features,
            'hit_group_onehot': hit_group_onehot
        }
        
        return features_dict, damage_value
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(self.run_dir, "checkpoints", filename)
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }, checkpoint_path)
        
        print(f"Checkpoint saved to {checkpoint_path}")


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train damage regression VAE model')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension size')
    parser.add_argument('--latent_dim', type=int, default=16, help='Latent dimension size')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--train_split', type=float, default=0.8, help='Training split')
    parser.add_argument('--num_epoch', type=int, default=20, help='Number of epochs')
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--log_interval', type=int, default=10, help='Log interval')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--positive_samples_file', type=str, default='../data/positive_samples.npy',
                       help='File containing precomputed positive samples')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--use_xavier_init', type=bool, default=True, help='Use Xavier initialization')
    parser.add_argument('--use_batchnorm', type=bool, default=True, help='Use batch normalization')
    parser.add_argument('--gradient_clipping', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--use_gradient_clipping', type=bool, default=False, help='Use gradient clipping')
    parser.add_argument('--kl_weight', type=float, default=1, help='Weight for KL divergence term')
    parser.add_argument('--num_samples', type=int, default=3000, help='Number of samples for prediction')
    parser.add_argument('--generate_uncertainty_plots', type=bool, default=True, 
                        help='Generate plots showing prediction uncertainty')
    return parser.parse_args()


def main():
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create run directory
    run_dir = create_run_dir()
    
    # Prepare datasets
    from main import prepare_datasets
    train_dataset, val_dataset = prepare_datasets(args)
    
    # Create model
    model = DamageRegressionVAE(
        num_maps=len(train_dataset.map_names),
        num_weapons=len(train_dataset.weapons),
        num_hit_groups=len(train_dataset.hit_group_names),
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
        use_xavier_init=args.use_xavier_init,
        use_batchnorm=args.use_batchnorm
    ).to(device)
    
    # Initialize wandb if enabled
    if args.use_wandb:
        import wandb
        wandb.init(
            project="damage-regression-vae-model",
            config=vars(args),
            name=f"vae_model_h{args.hidden_dim}_l{args.latent_dim}_kl{args.kl_weight}"
        )
        # wandb.watch(model, log="all", log_freq=100)
    
    # Create trainer and train
    trainer = VAERegressionTrainer(
        model=model,
        run_dir=run_dir,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        args=args
    )
    
    trainer.train()
    
    # After training, we don't need to generate uncertainty plots here anymore
    # since they're generated after each validation epoch
    print("Training completed!")


if __name__ == "__main__":
    main() 