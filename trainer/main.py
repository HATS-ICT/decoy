import os   
import json
import torch
import matplotlib.pyplot as plt
import random
import numpy as np
from dataset import DamageOutcomeDataset
from model import DamageOutcomeModel
from trainer import Trainer
from utils import create_run_dir
import argparse


def set_seed(seed):
    """
    Set random seed for reproducibility across all libraries
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Using random seed: {seed}")

def validate_settings(args):
    """
    Validate that the model settings are correct
    
    Args:
        args: Command line arguments
    """
    # Check that loss weights sum to 1
    loss_weights_sum = (
        args.damage_indicator_loss_weight + 
        args.damage_value_loss_weight + 
        args.hit_group_loss_weight
    )
    
    assert abs(loss_weights_sum - 1.0) < 1e-6, (
        f"Loss weights must sum to 1.0, but got {loss_weights_sum} "
        f"(damage_indicator: {args.damage_indicator_loss_weight}, "
        f"damage_value: {args.damage_value_loss_weight}, "
        f"hit_group: {args.hit_group_loss_weight})"
    )

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train damage outcome model')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension size')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--train_split', type=float, default=0.8, help='Training split')
    parser.add_argument('--num_epoch', type=int, default=5, help='Number of epochs')
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--log_interval', type=int, default=10, help='Log interval')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--positive_samples_file', type=str, default='../data/positive_samples.npy',
                       help='File containing precomputed positive samples')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--damage_indicator_loss_weight', type=float, default=0.2, help='Damage indicator loss weight')
    parser.add_argument('--damage_value_loss_weight', type=float, default=0.4, help='Damage value loss weight')
    parser.add_argument('--hit_group_loss_weight', type=float, default=0.4, help='Hit group loss weight')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--use_xavier_init', type=bool, default=True, help='Use Xavier initialization')
    parser.add_argument('--use_batchnorm', type=bool, default=True, help='Use batch normalization')
    parser.add_argument('--gradient_clipping', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--use_gradient_clipping', type=bool, default=False, help='Use gradient clipping')
    parser.add_argument('--weighted_cross_entropy', type=bool, default=True, help='Use weighted cross entropy')
    parser.add_argument('--weighted_mse', type=bool, default=True, help='Use weighted mse')
    parser.add_argument('--model_mode', type=str, default='joint', help='Mode to use for the model')
    return parser.parse_args()

def prepare_datasets(args):
    data_dir = os.path.join("..", "data", "player_seq_allmap_full_npz_damage")
    file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')]
    random.shuffle(file_paths)
    train_file_paths = file_paths[:int(len(file_paths) * args.train_split)]
    val_file_paths = file_paths[int(len(file_paths) * args.train_split):]

    train_positive_samples, val_positive_samples = load_positive_samples(args)
    
    train_dataset = DamageOutcomeDataset(
        train_file_paths, 
        positive_samples=train_positive_samples,
        num_workers=args.num_workers
    )
    
    val_dataset = DamageOutcomeDataset(
        val_file_paths, 
        positive_samples=val_positive_samples,
        num_workers=args.num_workers
    )
    
    return train_dataset, val_dataset

def load_positive_samples(args):
    positive_samples_file = args.positive_samples_file if os.path.exists(args.positive_samples_file) else None
    train_positive_samples = None
    val_positive_samples = None
    
    if positive_samples_file:
        positive_samples = np.load(positive_samples_file)
        shuffle_indices = np.random.permutation(len(positive_samples))
        split_idx = int(len(positive_samples) * args.train_split)
        train_positive_samples = positive_samples[shuffle_indices[:split_idx]]
        val_positive_samples = positive_samples[shuffle_indices[split_idx:]]
    
    return train_positive_samples, val_positive_samples

def init_logging(model, run_dir, device, args):
    config = vars(args)
    config.update({
        'model_architecture': str(model),
        'device': str(device)
    })

    with open(os.path.join(run_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    if args.use_wandb:
        import wandb
        experiment_name = f"damage_model_{args.hidden_dim}_{args.learning_rate}"
        wandb.init(
            project="damage-outcome-prediction-hitgroup",
            config=config,
            name=experiment_name
        )
        # wandb.watch(model, log="all", log_freq=100)

def main():
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    set_seed(args.seed)
    validate_settings(args)
    
    train_dataset, val_dataset = prepare_datasets(args)
    model = DamageOutcomeModel(
        num_maps=len(train_dataset.map_names),
        num_weapons=len(train_dataset.weapons),
        num_hit_groups=len(train_dataset.hit_group_names),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        use_xavier_init=args.use_xavier_init,
        use_batchnorm=args.use_batchnorm,
        mode=args.model_mode
    ).to(device)

    run_dir = create_run_dir()

    init_logging(model, run_dir, device, args)
    
    trainer = Trainer(
        model=model,
        run_dir=run_dir,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        args=args
    )
    trainer.train()
    print("Training completed and model saved!")

if __name__ == "__main__":
    main()