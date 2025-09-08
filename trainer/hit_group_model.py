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
from utils import create_run_dir, calculate_multiclass_classification_metrics, plot_confusion_matrix, plot_class_metrics
import matplotlib.pyplot as plt
import wandb
from dataset import HIT_GROUP_NAMES


class HitGroupClassificationModel(nn.Module):
    def __init__(self, num_maps, num_weapons, num_hit_groups, 
                 hidden_dim=128, dropout=0.3, use_xavier_init=False, use_batchnorm=False):
        """
        Classification-only model for hit group prediction
        
        Args:
            num_maps: Number of map features
            num_weapons: Number of weapon features
            num_hit_groups: Number of hit group categories
            hidden_dim: Hidden dimension size
            dropout: Dropout rate
            use_xavier_init: Whether to use Xavier (Glorot) initialization
            use_batchnorm: Whether to use batch normalization
        """
        super(HitGroupClassificationModel, self).__init__()
        
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
        hit_group_logits = self.hit_group_classifier(latent_features)
        
        return hit_group_logits


class HitGroupTrainer:
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
            print(f"Train Accuracy: {train_metrics['accuracy']:.4f}, Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"Train F1 (macro): {train_metrics['f1_macro']:.4f}, Val F1 (macro): {val_metrics['f1_macro']:.4f}")
        
        if self.args.use_wandb:
            wandb.finish()
        return
    
    def _log_metrics(self, train_metrics, val_metrics):
        # Log to wandb if enabled
        if self.args.use_wandb:
            wandb.log({
                'epoch': self.current_epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_accuracy': train_metrics['accuracy'],
                'train_precision_macro': train_metrics['precision_macro'],
                'train_recall_macro': train_metrics['recall_macro'],
                'train_f1_macro': train_metrics['f1_macro'],
                'train_top_2_accuracy': train_metrics['top_2_accuracy'],
                'train_top_3_accuracy': train_metrics['top_3_accuracy'],
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
                'val_precision_macro': val_metrics['precision_macro'],
                'val_recall_macro': val_metrics['recall_macro'],
                'val_f1_macro': val_metrics['f1_macro'],
                'val_top_2_accuracy': val_metrics['top_2_accuracy'],
                'val_top_3_accuracy': val_metrics['top_3_accuracy'],
            }, step=self.global_step)
        
        # Save plots
        self._save_plots(train_metrics, 'train')
        self._save_plots(val_metrics, 'val')
    
    def _save_plots(self, metrics, prefix):
        plots_dir = os.path.join(self.run_dir, "plots")
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        # Create confusion matrix plot
        cm_path = os.path.join(plots_dir, f"{prefix}_confusion_matrix_epoch_{self.current_epoch+1}.png")
        plot_confusion_matrix(
            np.array(metrics['confusion_matrix']),
            HIT_GROUP_NAMES,
            f"{prefix.capitalize()} Hit Group Confusion Matrix (Epoch {self.current_epoch+1})",
            cm_path
        )
        
        # Create class metrics plot
        class_metrics_path = os.path.join(plots_dir, f"{prefix}_class_metrics_epoch_{self.current_epoch+1}.png")
        plot_class_metrics(
            metrics,
            HIT_GROUP_NAMES,
            f"{prefix.capitalize()} Hit Group Class Metrics (Epoch {self.current_epoch+1})",
            class_metrics_path
        )
        
        if self.args.use_wandb:
            wandb.log({
                f'{prefix}_confusion_matrix': wandb.Image(cm_path),
                f'{prefix}_class_metrics': wandb.Image(class_metrics_path)
            }, step=self.global_step)
    
    def train_epoch(self):
        self.model.train()
        return self._run_epoch(self.train_dataset, is_training=True)

    def validate_epoch(self):
        self.model.eval()
        return self._run_epoch(self.val_dataset, is_training=False)
    
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
            collate_fn=self._classification_collate_fn
        )
        
        total_loss = 0.0
        all_logits = []
        all_targets = []
        
        pbar = tqdm(data_loader, desc=f"Epoch {self.current_epoch+1}/{self.args.num_epoch} [{mode}]")
        
        if not is_training:
            # Use torch.no_grad() for validation
            with torch.no_grad():
                for batch_x, batch_y in pbar:
                    loss, logits, targets = self._process_batch(batch_x, batch_y, is_training)
                    total_loss += loss.item() * len(targets)
                    all_logits.append(logits.detach().cpu().numpy())
                    all_targets.append(targets.detach().cpu().numpy())
                    pbar.set_postfix(loss=loss.item())
        else:
            for batch_x, batch_y in pbar:
                loss, logits, targets = self._process_batch(batch_x, batch_y, is_training)
                total_loss += loss.item() * len(targets)
                all_logits.append(logits.detach().cpu().numpy())
                all_targets.append(targets.detach().cpu().numpy())
                pbar.set_postfix(loss=loss.item())
                self.global_step += 1
        
        # Convert to numpy arrays
        all_logits = np.vstack(all_logits)
        all_targets = np.concatenate(all_targets)
        
        # Calculate metrics
        metrics, preds = calculate_multiclass_classification_metrics(all_targets, all_logits)
        metrics['loss'] = total_loss / len(all_targets)
        metrics['logits'] = all_logits
        metrics['targets'] = all_targets
        metrics['predictions'] = preds
        
        return total_loss / len(all_targets), metrics
    
    def _process_batch(self, batch_x, batch_y, is_training=True):
        """Process a single batch"""
        # Move data to device
        batch_x = {k: v.to(self.device) for k, v in batch_x.items()}
        batch_y = batch_y.to(self.device)
        
        if is_training:
            self.optimizer.zero_grad()
        
        # Forward pass
        logits = self.model(batch_x)
        
        # Calculate loss
        if self.args.weighted_cross_entropy:
            weights = self.train_dataset.hit_group_class_weights.to(self.device)
            loss = F.cross_entropy(logits, batch_y, weight=weights)
        else:
            loss = F.cross_entropy(logits, batch_y)
        
        if is_training:
            loss.backward()
            if self.args.use_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gradient_clipping)
            self.optimizer.step()
            
            if self.args.use_wandb and self.global_step % self.args.log_interval == 0:
                wandb.log({'train_batch_loss': loss.item()}, step=self.global_step)
        
        return loss, logits, batch_y
    
    def _classification_collate_fn(self, batch):
        """Custom collate function for classification model that only uses samples with damage"""
        # Filter out samples without damage
        filtered_batch = [item for item in batch if item['damage_indicator'].item() > 0]
        
        if not filtered_batch:
            # If no samples with damage, return empty tensors
            return {
                'map': torch.empty((0, len(self.train_dataset.map_names))),
                'coords_and_angles': torch.empty((0, 11)),
                'weapon': torch.empty((0, len(self.train_dataset.weapons))),
                'armor_features': torch.empty((0, 2)),
            }, torch.empty((0,), dtype=torch.long)
        
        # Stack features
        maps = torch.stack([item['map'] for item in filtered_batch])
        coords_and_angles = torch.stack([item['coords_and_angles'] for item in filtered_batch])
        weapons = torch.stack([item['weapon'] for item in filtered_batch])
        armor_features = torch.stack([item['armor_features'] for item in filtered_batch])
        hit_group = torch.stack([item['hit_group'] for item in filtered_batch]).squeeze()
        
        # Group all X features in a dictionary
        features_dict = {
            'map': maps,
            'coords_and_angles': coords_and_angles,
            'weapon': weapons,
            'armor_features': armor_features,
        }
        
        return features_dict, hit_group
    
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
    parser = argparse.ArgumentParser(description='Train hit group classification model')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension size')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--train_split', type=float, default=0.8, help='Training split')
    parser.add_argument('--num_epoch', type=int, default=5, help='Number of epochs')
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
    parser.add_argument('--weighted_cross_entropy', type=bool, default=True, help='Use weighted cross entropy loss')
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
    model = HitGroupClassificationModel(
        num_maps=len(train_dataset.map_names),
        num_weapons=len(train_dataset.weapons),
        num_hit_groups=len(train_dataset.hit_group_names),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        use_xavier_init=args.use_xavier_init,
        use_batchnorm=args.use_batchnorm
    ).to(device)
    
    # Initialize wandb if enabled
    if args.use_wandb:
        import wandb
        wandb.init(
            project="hit-group-classification-model",
            config=vars(args),
            name=f"hit_group_model_{args.hidden_dim}_{args.learning_rate}"
        )
        wandb.watch(model, log="all", log_freq=100)
    
    # Create trainer and train
    trainer = HitGroupTrainer(
        model=model,
        run_dir=run_dir,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        args=args
    )
    
    trainer.train()
    print("Training completed!")


if __name__ == "__main__":
    main() 