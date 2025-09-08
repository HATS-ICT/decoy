import os
import time
import uuid
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           confusion_matrix, roc_curve, auc, mean_absolute_error,
                           mean_squared_error, r2_score)
import seaborn as sns
import wandb
from dataset import HIT_GROUP_NAMES, WEAPON_NAMES  
import pandas as pd

def create_run_dir(base_dir="logs"):
    """Create a unique directory for the current training run."""
    # Create base log directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Create a unique run ID using timestamp and random UUID
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    run_id = f"{timestamp}-{str(uuid.uuid4())[:5]}"
    
    # Create the run directory
    run_dir = os.path.join(base_dir, run_id)
    os.makedirs(run_dir)
    
    # Create subdirectories for different types of logs
    os.makedirs(os.path.join(run_dir, "checkpoints"))
    os.makedirs(os.path.join(run_dir, "plots"))
    return run_dir

def calculate_binary_classification_metrics(targets, probs, find_optimal_threshold=True):
    """Calculate metrics for binary classification with threshold optimization."""
    base_metrics = {}
    
    # Default threshold metrics
    preds = (probs > 0.5).astype(int)
    base_metrics.update({
        'accuracy': float(accuracy_score(targets, preds)),
        'precision': float(precision_score(targets, preds, zero_division=0)),
        'recall': float(recall_score(targets, preds, zero_division=0)),
        'f1_score': float(f1_score(targets, preds, zero_division=0)),
        'confusion_matrix': confusion_matrix(targets, preds).tolist(),
    })
    
    # ROC and AUC
    fpr, tpr, thresholds = roc_curve(targets, probs)
    base_metrics['roc_auc'] = float(auc(fpr, tpr))
    
    # Optimal threshold search
    if find_optimal_threshold:
        optimal_threshold = 0
        max_f1 = 0
        
        thresholds = np.linspace(0, 1, 100)
        for threshold in thresholds:
            threshold_preds = (probs > threshold).astype(int)
            f1 = f1_score(targets, threshold_preds, zero_division=0)
            if f1 > max_f1:
                max_f1 = f1
                optimal_threshold = threshold
        
        opt_preds = (probs > optimal_threshold).astype(int)
        base_metrics.update({
            'optimal_threshold': float(optimal_threshold),
            'optimal_f1_score': float(max_f1),
            'optimal_precision': float(precision_score(targets, opt_preds, zero_division=0)),
            'optimal_recall': float(recall_score(targets, opt_preds, zero_division=0)),
            'optimal_accuracy': float(accuracy_score(targets, opt_preds))
        })
    
    return base_metrics, preds, opt_preds

def calculate_multiclass_classification_metrics(targets, logits):
    """Calculate metrics for multiclass classification including top-k accuracy."""
    # Top-1 accuracy (standard)
    preds = np.argmax(logits, axis=1)
    
    metrics = {
        'accuracy': float(accuracy_score(targets, preds)),
        'precision_macro': float(precision_score(targets, preds, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(targets, preds, average='macro', zero_division=0)),
        'f1_macro': float(f1_score(targets, preds, average='macro', zero_division=0)),
        'confusion_matrix': confusion_matrix(targets, preds).tolist(),
    }

    # assert len(metrics['confusion_matrix']) == len(HIT_GROUP_NAMES)
    
    # Per-class metrics
    metrics.update({
        'per_class_precision': precision_score(targets, preds, average=None, zero_division=0).tolist(),
        'per_class_recall': recall_score(targets, preds, average=None, zero_division=0).tolist(),
        'per_class_f1': f1_score(targets, preds, average=None, zero_division=0).tolist()
    })
    
    # Top-k accuracy
    for k in [2, 3]:
        if logits.shape[1] >= k:  # Only if we have enough classes
            # For each sample, check if true class is in top k predictions
            top_k_preds = np.argsort(logits, axis=1)[:, -k:]
            top_k_correct = 0
            
            for i, target in enumerate(targets):
                if target in top_k_preds[i]:
                    top_k_correct += 1
            
            metrics[f'top_{k}_accuracy'] = float(top_k_correct / len(targets))
    
    return metrics, preds

def calculate_regression_metrics(targets, preds):
    """Calculate metrics for regression."""
    mae = mean_absolute_error(targets, preds)
    mse = mean_squared_error(targets, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, preds)
    
    return {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'r2_score': float(r2)
    }


def evaluate_predictions(damage_indicator_probs, damage_value, hit_group_logits, 
                         damage_indicator_targets, damage_value_targets, hit_group_targets):
    """Evaluate predictions for damage outcome model."""
    damage_mask = damage_indicator_targets.squeeze() > 0
    # Calculate metrics for damage indicator prediction (binary)
    damage_indicator_metrics, damage_indicator_preds_default, damage_indicator_preds_optimal = calculate_binary_classification_metrics(
        damage_indicator_targets, damage_indicator_probs)
    
    # Calculate metrics for damage value prediction (regression)
    damage_value_metrics = calculate_regression_metrics(
        damage_value_targets[damage_mask], damage_value[damage_mask])
    
    # Calculate metrics for hit group prediction (multi-class)
    hit_group_metrics, hit_group_preds = calculate_multiclass_classification_metrics(
        hit_group_targets[damage_mask], hit_group_logits[damage_mask])
    
    metrics = {
        'damage_indicator_metrics': damage_indicator_metrics,
        'damage_value_metrics': damage_value_metrics,
        'hit_group_metrics': hit_group_metrics
    }

    predictions = {
        'damage_indicator_preds_default': damage_indicator_preds_default,
        'damage_indicator_preds_optimal': damage_indicator_preds_optimal,
        'hit_group_preds': hit_group_preds
    }
    
    return metrics, predictions


def log_metrics(metrics, epoch, global_step, run_dir, log_wandb=False, prefix='train'):
    """
    Log metrics to a JSON file in the run directory, indexed by epoch.
    
    Args:
        metrics (dict): Dictionary containing metrics to log
        epoch (int): Current epoch number
        run_dir (str): Directory to save the logs
        prefix (str): Prefix for the log file name (train/val)
    """
    json_path = os.path.join(run_dir, f"{prefix}_metrics.json")
    
    existing_data = {}
    if os.path.isfile(json_path):
        try:
            with open(json_path, 'r') as jsonfile:
                existing_data = json.load(jsonfile)
        except json.JSONDecodeError:
            existing_data = {}

    log_metrics_dict = {
        f"{prefix}-epoch": epoch,
        f"{prefix}-epoch-loss/total_loss": metrics['loss']['total_loss'],
        f"{prefix}-epoch-loss/damage_indicator_loss": metrics['loss']['damage_indicator_loss'],
        f"{prefix}-epoch-loss/damage_value_loss": metrics['loss']['damage_value_loss'],
        f"{prefix}-epoch-loss/hit_group_loss": metrics['loss']['hit_group_loss'],
        f"{prefix}-epoch-damage_indicator/accuracy": metrics['damage_indicator_metrics']['accuracy'],
        f"{prefix}-epoch-damage_indicator/precision": metrics['damage_indicator_metrics']['precision'],
        f"{prefix}-epoch-damage_indicator/recall": metrics['damage_indicator_metrics']['recall'],
        f"{prefix}-epoch-damage_indicator/f1_score": metrics['damage_indicator_metrics']['f1_score'],
        f"{prefix}-epoch-damage_indicator/roc_auc": metrics['damage_indicator_metrics']['roc_auc'],
        f"{prefix}-epoch-damage_indicator/optimal_threshold": metrics['damage_indicator_metrics']['optimal_threshold'],
        f"{prefix}-epoch-damage_indicator/optimal_f1_score": metrics['damage_indicator_metrics']['optimal_f1_score'],
        f"{prefix}-epoch-damage_indicator/optimal_precision": metrics['damage_indicator_metrics']['optimal_precision'],
        f"{prefix}-epoch-damage_indicator/optimal_recall": metrics['damage_indicator_metrics']['optimal_recall'],
        f"{prefix}-epoch-damage_indicator/optimal_accuracy": metrics['damage_indicator_metrics']['optimal_accuracy'],
        f"{prefix}-epoch-damage_value/mae": metrics['damage_value_metrics']['mae'],
        f"{prefix}-epoch-damage_value/mse": metrics['damage_value_metrics']['mse'],
        f"{prefix}-epoch-damage_value/rmse": metrics['damage_value_metrics']['rmse'],
        f"{prefix}-epoch-damage_value/r2_score": metrics['damage_value_metrics']['r2_score'],
        f"{prefix}-epoch-hit_group/accuracy": metrics['hit_group_metrics']['accuracy'],
        f"{prefix}-epoch-hit_group/precision_macro": metrics['hit_group_metrics']['precision_macro'],
        f"{prefix}-epoch-hit_group/recall_macro": metrics['hit_group_metrics']['recall_macro'],
        f"{prefix}-epoch-hit_group/f1_macro": metrics['hit_group_metrics']['f1_macro'], 
        f"{prefix}-epoch-hit_group/top_2_accuracy": metrics['hit_group_metrics']['top_2_accuracy'],
        f"{prefix}-epoch-hit_group/top_3_accuracy": metrics['hit_group_metrics']['top_3_accuracy'],
    }
    
    existing_data[str(epoch)] = log_metrics_dict
    
    # Write to JSON file
    with open(json_path, 'w') as jsonfile:
        json.dump(existing_data, jsonfile, indent=2)

    if log_wandb:
        wandb.log(log_metrics_dict, step=global_step)

def plot_confusion_matrix(cm, class_names, title, save_path):
    """Plot confusion matrix as a heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path

def plot_roc_curve(fpr, tpr, roc_auc, title, save_path):
    """Plot ROC curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path

def plot_regression_scatter(y_true, y_pred, r2, title, save_path):
    """Plot regression scatter plot with true vs predicted values."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.1)
    
    # Add perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{title} (R² = {r2:.3f})')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path

def plot_class_metrics(metrics, class_names, title, save_path):
    """Plot per-class metrics as a bar chart."""
    plt.figure(figsize=(12, 6))
    
    # Get the actual number of classes from the metrics
    num_classes = len(metrics['per_class_precision'])
    
    x = np.arange(num_classes)
    width = 0.25
    
    plt.bar(x - width, metrics['per_class_precision'], width, label='Precision')
    plt.bar(x, metrics['per_class_recall'], width, label='Recall')
    plt.bar(x + width, metrics['per_class_f1'], width, label='F1 Score')
    
    plt.xlabel('Classes')
    plt.ylabel('Score')
    plt.title(title)
    plt.xticks(x, class_names, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path

def plot_loss_history(loss_data, title, save_path):
    """Plot loss history."""
    plt.figure(figsize=(10, 6))
    for loss_name, loss_values in loss_data.items():
        plt.plot(loss_values, label=loss_name)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_damage_distribution_matrix(targets, predictions, weapon_indices, body_part_indices, plot_path):
    """
    Plots a matrix of histograms showing the distribution of predicted vs. true damage values
    conditioned on weapon type and body part hit.

    Args:
        targets (np.array): Ground truth damage values (shape: N, 1).
        predictions (np.array): Predicted damage values (shape: N, 1).
        weapon_indices (np.array): Weapon type indices (shape: N, 1).
        body_part_indices (np.array): Body part hit indices (shape: N, 1).
        plot_path (str): Path to save the generated plot.
    """
    from scipy.spatial.distance import jensenshannon
    import matplotlib.pyplot as plt
    import numpy as np

    body_part_labels = HIT_GROUP_NAMES
    weapon_labels = WEAPON_NAMES

    selected_weapons = (0, 12, 2, 10, 32)
    selected_body_parts = (0, 2, 3, 4, 5)
    
    # Filter data to only include selected weapons and body parts
    mask = np.zeros_like(targets, dtype=bool)
    for weapon_idx in selected_weapons:
        for body_part_idx in selected_body_parts:
            weapon_mask = weapon_indices == weapon_idx
            body_part_mask = body_part_indices == body_part_idx
            mask = mask | (weapon_mask & body_part_mask)
    
    targets = targets[mask]
    predictions = predictions[mask]
    weapon_indices = weapon_indices[mask]
    body_part_indices = body_part_indices[mask]
        
    # Define color scheme
    true_color = "#1f77b4"  # Blue
    pred_color = "#ff7f0e"  # Orange
    overlap_color = "#9467bd"  # Purple

    # Compute overall mean JSD for selected weapons and body parts only
    total_jsd = 0
    count = 0

    sample_size = 3000
    for weapon_idx in selected_weapons:
        for body_part_idx in selected_body_parts:
            mask = (weapon_indices == weapon_idx) & (body_part_indices == body_part_idx)
            true_data = targets[mask]
            pred_data = predictions[mask]

            if len(true_data) > sample_size:
                sample_indices = np.random.choice(len(true_data), sample_size, replace=False)
                true_data = true_data[sample_indices]
                pred_data = pred_data[sample_indices]

            if len(true_data) == 0 or len(pred_data) == 0:
                continue

            bins = np.linspace(0, 100, 21)
            true_hist, _ = np.histogram(true_data, bins=bins, density=True)
            pred_hist, _ = np.histogram(pred_data, bins=bins, density=True)

            true_hist += 1e-8
            pred_hist += 1e-8

            jsd = jensenshannon(true_hist, pred_hist)
            total_jsd += jsd
            count += 1

    mean_jsd = total_jsd / count if count > 0 else 0

    # Create a 5x5 matrix plot for selected weapons and body parts
    fig, axes = plt.subplots(len(selected_weapons), len(selected_body_parts), 
                             figsize=(15, 12), sharex=True, sharey=False)  # Changed sharey to False

    for i, weapon_idx in enumerate(selected_weapons):
        for j, body_part_idx in enumerate(selected_body_parts):
            ax = axes[i, j]
            weapon = weapon_labels[weapon_idx]
            body_part = body_part_labels[body_part_idx]

            # Filter data
            mask = (weapon_indices == weapon_idx) & (body_part_indices == body_part_idx)
            true_data = targets[mask]
            pred_data = predictions[mask]

            if len(true_data) == 0 or len(pred_data) == 0:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.text(0.5, 0.5, "No Data", ha='center', va='center')
                continue

            bins = np.linspace(0, 100, 21)
            true_hist, _ = np.histogram(true_data, bins=bins, density=True)
            pred_hist, _ = np.histogram(pred_data, bins=bins, density=True)

            true_hist += 1e-8
            pred_hist += 1e-8

            jsd = jensenshannon(true_hist, pred_hist)

            # Plot histograms
            bin_centers = (bins[:-1] + bins[1:]) / 2
            ax.bar(bin_centers, true_hist, width=4, color=true_color, alpha=0.7)
            ax.bar(bin_centers, pred_hist, width=4, color=pred_color, alpha=0.7)
            overlap = np.minimum(true_hist, pred_hist)
            ax.bar(bin_centers, overlap, width=4, color=overlap_color, alpha=1)
            
            # Set y-axis limits based on the maximum value in this specific histogram
            max_height = max(np.max(true_hist), np.max(pred_hist)) * 1.1  # Add 10% padding
            ax.set_ylim(0, max_height)

            # Combine JSD and sample size in a single box at top right
            ax.text(0.95, 0.95, f"JSD: {jsd:.3f}\nn={min(len(true_data), sample_size)}", transform=ax.transAxes,
                    fontsize=8, ha='right', va='top',
                    bbox=dict(facecolor='white', alpha=0.6))

            # Add axis labels
            if i == len(selected_weapons) - 1:  # Bottom row
                ax.set_xlabel("Damage Value", fontsize=9)
            
            if j == 0:  # First column
                ax.set_ylabel(weapon, fontsize=10)
                
            # Add titles for columns and row labels
            if i == 0:
                ax.set_title(body_part, fontsize=10)

    # Add super title with Mean JSD
    fig.suptitle(f"Damage Distribution Alignment (Mean JSD = {mean_jsd:.3f})",
                 fontsize=14, fontweight='bold')

    # Create a single legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=true_color, alpha=0.7, label="Ground Truth"),
               plt.Rectangle((0, 0), 1, 1, color=pred_color, alpha=0.7, label="Predicted"),
               plt.Rectangle((0, 0), 1, 1, color=overlap_color, alpha=1, label="Overlap")]
    fig.legend(handles=handles, loc='upper right', fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit title

    # Save the figure
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)  # Close the figure to prevent display issues

    print(f"Plot saved to {plot_path}")


# def plot_damage_distribution_matrix(weapon_indices, hit_group_targets, damage_value_targets, 
#                                    damage_value_preds, damage_indicator_targets, hit_group_names, 
#                                    weapon_names, epoch, run_dir, prefix='train', log_wandb=False, 
#                                    global_step=None, x_range=(0, 100), fixed_bins=None):
#     """
#     Create a matrix plot showing damage distributions for different weapons and body parts.
    
#     Args:
#         weapon_indices: Indices of weapons for each sample
#         hit_group_targets: Target hit group indices
#         damage_value_targets: Target damage values
#         damage_value_preds: Predicted damage values
#         damage_indicator_targets: Binary indicators of whether damage occurred
#         hit_group_names: List of hit group names
#         weapon_names: List of weapon names
#         epoch: Current epoch number
#         run_dir: Directory to save the plot
#         prefix: Prefix for the plot file name (train/val)
#         log_wandb: Whether to log the plot to wandb
#         global_step: Current global step for wandb logging
#         x_range: Tuple of (min, max) for x-axis limits
#         fixed_bins: If provided, use this fixed number of bins instead of calculating
#     """
#     # Only consider samples where damage occurred
#     damage_mask = damage_indicator_targets.flatten() > 0
    
#     if not np.any(damage_mask):
#         print("No damage samples to plot distribution matrix")
#         return None
    
#     # Filter data by damage mask
#     filtered_weapon_indices = weapon_indices[damage_mask]
#     filtered_hit_group_targets = hit_group_targets[damage_mask]
#     filtered_damage_value_targets = damage_value_targets[damage_mask].flatten()
#     filtered_damage_value_preds = damage_value_preds[damage_mask].flatten()
    
#     # Adjust values at the upper boundary to avoid binning issues
#     epsilon = 1e-6
#     filtered_damage_value_targets = np.where(
#         filtered_damage_value_targets >= x_range[1], 
#         x_range[1] - epsilon, 
#         filtered_damage_value_targets
#     )
#     filtered_damage_value_preds = np.where(
#         filtered_damage_value_preds >= x_range[1], 
#         x_range[1] - epsilon, 
#         filtered_damage_value_preds
#     )
    
#     # Convert indices to names
#     weapon_names_list = []
#     for idx in filtered_weapon_indices:
#         if idx >= 0 and idx < len(weapon_names):
#             weapon_names_list.append(weapon_names[idx])
#         else:
#             weapon_names_list.append("Unknown")
            
#     hit_group_names_list = []
#     for idx in filtered_hit_group_targets:
#         if idx >= 0 and idx < len(hit_group_names):
#             hit_group_names_list.append(hit_group_names[idx])
#         else:
#             hit_group_names_list.append("Unknown")
    
#     # Create DataFrames for ground truth and predictions
#     ground_truth_data = []
#     predicted_data = []
    
#     for i in range(len(weapon_names_list)):
#         ground_truth_data.append({
#             "Weapon": weapon_names_list[i],
#             "Body Part": hit_group_names_list[i],
#             "Damage": filtered_damage_value_targets[i],
#             "Type": "Ground Truth"
#         })
        
#         predicted_data.append({
#             "Weapon": weapon_names_list[i],
#             "Body Part": hit_group_names_list[i],
#             "Damage": filtered_damage_value_preds[i],
#             "Type": "Predicted"
#         })
    
#     df_ground_truth = pd.DataFrame(ground_truth_data)
#     df_predicted = pd.DataFrame(predicted_data)
#     df_combined = pd.concat([df_ground_truth, df_predicted])
    
#     # Filter to include only the most common weapons and body parts for clarity
#     weapon_counts = df_combined['Weapon'].value_counts()
    
#     # Select top 6 weapons and all body parts (usually there are only 6)
#     top_weapons = weapon_counts.nlargest(6).index.tolist()
#     df_filtered = df_combined[df_combined['Weapon'].isin(top_weapons)]
    
#     # Get unique weapons and body parts for the grid
#     unique_weapons = sorted(df_filtered['Weapon'].unique())
#     unique_body_parts = sorted(df_filtered['Body Part'].unique())
    
#     # Create a figure with a grid of subplots
#     n_weapons = len(unique_weapons)
#     n_body_parts = len(unique_body_parts)
    
#     fig, axes = plt.subplots(n_weapons, n_body_parts, figsize=(n_body_parts*3.5, n_weapons*3), 
#                              constrained_layout=True)
    
#     # Create a custom legend
#     from matplotlib.lines import Line2D
#     legend_elements = [
#         Line2D([0], [0], color='blue', lw=2, label='Ground Truth'),
#         Line2D([0], [0], color='red', lw=2, label='Predicted')
#     ]
    
#     # Dictionary to store metrics for each weapon-body part combination
#     distribution_metrics = {}
    
#     # Plot each weapon-body part combination in its own subplot
#     for i, weapon in enumerate(unique_weapons):
#         for j, body_part in enumerate(unique_body_parts):
#             ax = axes[i, j] if n_weapons > 1 and n_body_parts > 1 else axes[i] if n_body_parts == 1 else axes[j] if n_weapons == 1 else axes
            
#             # Get data for this weapon-body part combination
#             mask = (df_filtered['Weapon'] == weapon) & (df_filtered['Body Part'] == body_part)
#             subset = df_filtered[mask]
            
#             if len(subset) == 0:
#                 ax.text(0.5, 0.5, "No Data", ha='center', va='center')
#                 ax.set_xticks([])
#                 ax.set_yticks([])
#                 continue
            
#             # Get ground truth and prediction data
#             gt_data = subset[subset['Type'] == 'Ground Truth']['Damage']
#             pred_data = subset[subset['Type'] == 'Predicted']['Damage']
            
#             # Count samples
#             n_gt_samples = len(gt_data)
#             n_pred_samples = len(pred_data)
            
#             # Create consistent bin edges if fixed_bins is provided
#             if fixed_bins is not None and x_range is not None:
#                 # Create evenly spaced bins across the x_range
#                 bin_edges = np.linspace(x_range[0], x_range[1], fixed_bins + 1)
                
#                 # Plot histograms with the same bin edges - use stat='count' to show actual counts
#                 if len(gt_data) > 0:
#                     sns.histplot(gt_data, ax=ax, color='blue', alpha=0.5, label='Ground Truth', 
#                                 kde=True, bins=bin_edges, stat='count')
                
#                 if len(pred_data) > 0:
#                     sns.histplot(pred_data, ax=ax, color='red', alpha=0.5, label='Predicted',
#                                 kde=True, bins=bin_edges, stat='count')
#             else:
#                 # Use automatic bin calculation if fixed_bins is not provided
#                 if len(gt_data) > 0:
#                     # Calculate appropriate bins
#                     gt_range = gt_data.max() - gt_data.min() if len(gt_data) > 1 else 1
                    
#                     # Calculate IQR safely
#                     iqr = np.percentile(gt_data, 75) - np.percentile(gt_data, 25) if len(gt_data) > 1 else 1
#                     # Avoid division by zero
#                     if iqr == 0:
#                         iqr = 1
                        
#                     # Calculate bin width using Freedman-Diaconis rule with safeguards
#                     bin_width = 2 * iqr / (len(gt_data) ** (1/3)) if len(gt_data) > 1 else 1
                    
#                     # Avoid division by zero
#                     if bin_width == 0:
#                         bin_width = 1
                        
#                     # Calculate number of bins
#                     bins = max(5, min(20, int(gt_range / bin_width) if gt_range > 0 else 5))
                    
#                     sns.histplot(gt_data, ax=ax, color='blue', alpha=0.5, label='Ground Truth', 
#                                 kde=True, bins=bins, stat='count')
                
#                 if len(pred_data) > 0:
#                     # Calculate appropriate bins
#                     pred_range = pred_data.max() - pred_data.min() if len(pred_data) > 1 else 1
                    
#                     # Calculate IQR safely
#                     iqr = np.percentile(pred_data, 75) - np.percentile(pred_data, 25) if len(pred_data) > 1 else 1
#                     # Avoid division by zero
#                     if iqr == 0:
#                         iqr = 1
                        
#                     # Calculate bin width using Freedman-Diaconis rule with safeguards
#                     bin_width = 2 * iqr / (len(pred_data) ** (1/3)) if len(pred_data) > 1 else 1
                    
#                     # Avoid division by zero
#                     if bin_width == 0:
#                         bin_width = 1
                        
#                     # Calculate number of bins
#                     bins = max(5, min(20, int(pred_range / bin_width) if pred_range > 0 else 5))
                    
#                     sns.histplot(pred_data, ax=ax, color='red', alpha=0.5, label='Predicted',
#                                 kde=True, bins=bins, stat='count')
            
#             # Calculate distribution metrics if we have enough samples
#             if len(gt_data) > 5 and len(pred_data) > 5:
#                 metrics = calculate_distribution_metrics(
#                     gt_data.values, pred_data.values, 
#                     bins=fixed_bins if fixed_bins is not None else 10,
#                     range=x_range
#                 )
#                 distribution_metrics[f"{weapon}_{body_part}"] = metrics
                
#                 # Add distribution metrics to the plot
#                 if not np.isnan(metrics['histogram_intersection']):
#                     metrics_text = (
#                         f"Overlap: {metrics['histogram_intersection']:.2f}\n"
#                         f"EMD: {metrics['wasserstein_distance']:.2f}\n"
#                         f"KS: {metrics['ks_statistic']:.2f}"
#                     )
#                     ax.text(0.95, 0.95, metrics_text, 
#                             transform=ax.transAxes, 
#                             verticalalignment='top', 
#                             horizontalalignment='right',
#                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
#             # Add sample count to the plot
#             ax.text(0.05, 0.95, f"GT: {n_gt_samples} samples\nPred: {n_pred_samples} samples", 
#                     transform=ax.transAxes, 
#                     verticalalignment='top', 
#                     horizontalalignment='left',
#                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
#             # Set title and labels
#             ax.set_title(f"{weapon} - {body_part}")
            
#             # Set x-axis limits if provided
#             if x_range:
#                 ax.set_xlim(x_range)
            
#             # Only set x and y labels for the bottom and left subplots
#             if i == n_weapons - 1:
#                 ax.set_xlabel("Damage")
#             else:
#                 ax.set_xlabel("")
                
#             if j == 0:
#                 ax.set_ylabel("Count")
#             else:
#                 ax.set_ylabel("")
            
#             # Let each subplot determine its own appropriate limits for y-axis
#             ax.autoscale(axis='y')
    
#     # Add a common legend at the bottom
#     fig.legend(handles=legend_elements, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0))
    
#     # Add a title for the entire figure
#     fig.suptitle(f"{prefix.capitalize()} Damage Distribution Matrix (Epoch {epoch})", fontsize=16, y=1.02)
    
#     # Save the plot
#     plots_dir = os.path.join(run_dir, "plots")
#     epoch_plots_dir = os.path.join(plots_dir, f"epoch_{epoch}")
#     if not os.path.exists(epoch_plots_dir):
#         os.makedirs(epoch_plots_dir)
    
#     plot_path = os.path.join(epoch_plots_dir, f"{prefix}_damage_distribution_matrix.png")
#     plt.savefig(plot_path, bbox_inches='tight', dpi=300)
#     plt.close()
    
#     # Calculate overall metrics across all data
#     if len(filtered_damage_value_targets) > 10 and len(filtered_damage_value_preds) > 10:
#         overall_metrics = calculate_distribution_metrics(
#             filtered_damage_value_targets, 
#             filtered_damage_value_preds,
#             bins=fixed_bins if fixed_bins is not None else 10,
#             range=x_range
#         )
#     else:
#         overall_metrics = {
#             'kl_divergence': float('nan'),
#             'js_divergence': float('nan'),
#             'wasserstein_distance': float('nan'),
#             'histogram_intersection': float('nan'),
#             'ks_statistic': float('nan'),
#             'ks_pvalue': float('nan')
#         }
    
#     # Log metrics to wandb
#     if log_wandb and global_step is not None:
#         wandb.log({
#             f"plots/{prefix}_damage_distribution_matrix": wandb.Image(plot_path),
#             f"distribution_metrics/{prefix}_overall_histogram_intersection": overall_metrics['histogram_intersection'],
#             f"distribution_metrics/{prefix}_overall_wasserstein_distance": overall_metrics['wasserstein_distance'],
#             f"distribution_metrics/{prefix}_overall_kl_divergence": overall_metrics['kl_divergence'],
#             f"distribution_metrics/{prefix}_overall_js_divergence": overall_metrics['js_divergence'],
#             f"distribution_metrics/{prefix}_overall_ks_statistic": overall_metrics['ks_statistic'],
#             f"distribution_metrics/{prefix}_overall_ks_pvalue": overall_metrics['ks_pvalue']
#         }, step=global_step)
    
#     return plot_path, distribution_metrics, overall_metrics

def plot_metrics(results, epoch, global_step, run_dir, prefix='train', log_wandb=False):
    """
    Create and save plots for the metrics, and optionally log them to wandb.
    
    Args:
        results (dict): Dictionary containing metrics and predictions to visualize
        epoch (int): Current epoch number
        global_step (int): Current global step
        run_dir (str): Directory to save the plots
        prefix (str): Prefix for the plot file names (train/val)
        log_wandb (bool): Whether to log plots to wandb
    """
    # Create base plots directory
    plots_dir = os.path.join(run_dir, "plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Create epoch-specific directory
    epoch_plots_dir = os.path.join(plots_dir, f"epoch_{epoch}")
    if not os.path.exists(epoch_plots_dir):
        os.makedirs(epoch_plots_dir)
    
    # Extract metrics and predictions from results
    metrics = results['metrics']
    predictions = results['predictions']
    
    # Dictionary to store paths to generated plots for wandb logging
    plot_paths = {}
    
    # 1. Plot confusion matrices
    # Binary classification confusion matrix
    binary_cm = np.array(metrics['damage_indicator_metrics']['confusion_matrix'])
    binary_cm_path = os.path.join(epoch_plots_dir, f"{prefix}_binary_confusion_matrix.png")
    plot_paths['binary_cm'] = plot_confusion_matrix(
        binary_cm, 
        ['No Damage', 'Damage'], 
        f"{prefix.capitalize()} Damage Indicator Confusion Matrix (Epoch {epoch})",
        binary_cm_path
    )
    
    # Multiclass confusion matrix
    multiclass_cm = np.array(metrics['hit_group_metrics']['confusion_matrix'])
    multiclass_cm_path = os.path.join(epoch_plots_dir, f"{prefix}_multiclass_confusion_matrix.png")
    # Use actual hit group names from dataset
    plot_paths['multiclass_cm'] = plot_confusion_matrix(
        multiclass_cm,
        HIT_GROUP_NAMES,
        f"{prefix.capitalize()} Hit Group Confusion Matrix (Epoch {epoch})",
        multiclass_cm_path
    )
    
    # 2. Plot ROC curve for binary classification
    # Calculate ROC curve from the raw predictions and targets
    damage_indicator_targets = predictions['damage_indicator_targets'].flatten()
    damage_indicator_probs = predictions['damage_indicator_probs'].flatten()
    
    fpr, tpr, _ = roc_curve(damage_indicator_targets, damage_indicator_probs)
    roc_auc = metrics['damage_indicator_metrics']['roc_auc']
    roc_path = os.path.join(epoch_plots_dir, f"{prefix}_roc_curve.png")
    plot_paths['roc_curve'] = plot_roc_curve(
        fpr, tpr, roc_auc,
        f"{prefix.capitalize()} Damage Indicator ROC Curve (Epoch {epoch})",
        roc_path
    )

    # 3. Plot per-class metrics for hit group classification
    class_metrics_path = os.path.join(epoch_plots_dir, f"{prefix}_class_metrics.png")
    plot_paths['class_metrics'] = plot_class_metrics(
        metrics['hit_group_metrics'],
        HIT_GROUP_NAMES,
        f"{prefix.capitalize()} Hit Group Class Metrics (Epoch {epoch})",
        class_metrics_path
    )
    
    # 4. Plot regression scatter plot for damage value prediction
    # Get only the samples where damage occurred for regression plot
    damage_mask = predictions['damage_indicator_targets'].flatten() > 0
    
    if np.any(damage_mask):
        damage_value_targets = predictions['damage_value_targets'].flatten()[damage_mask]
        damage_value_preds = predictions['damage_value'].flatten()[damage_mask]
        
        regression_path = os.path.join(epoch_plots_dir, f"{prefix}_regression_scatter.png")
        plot_paths['regression'] = plot_regression_scatter(
            damage_value_targets,
            damage_value_preds,
            metrics['damage_value_metrics']['r2_score'],
            f"{prefix.capitalize()} Damage Value Prediction (Epoch {epoch})",
            regression_path
        )
    
    # 5. Plot damage distribution matrix
    if 'weapon_indices' in predictions:
        distribution_matrix_path, distribution_metrics, overall_metrics = plot_damage_distribution_matrix(
            predictions['weapon_indices'],
            predictions['hit_group_targets'],
            predictions['damage_value_targets'],
            predictions['damage_value'],
            predictions['damage_indicator_targets'],
            HIT_GROUP_NAMES,
            WEAPON_NAMES,
            epoch,
            run_dir,
            prefix=prefix,
            log_wandb=log_wandb,
            global_step=global_step
        )
        if distribution_matrix_path:
            plot_paths['damage_distribution_matrix'] = distribution_matrix_path
    
    # Log plots to wandb if enabled
    if log_wandb:
        # Create a dictionary of images under the "plots" tag
        plots = {
            f"plots/{prefix}_binary_confusion_matrix": wandb.Image(plot_paths['binary_cm']),
            f"plots/{prefix}_multiclass_confusion_matrix": wandb.Image(plot_paths['multiclass_cm']),
            f"plots/{prefix}_roc_curve": wandb.Image(plot_paths['roc_curve']),
            f"plots/{prefix}_class_metrics": wandb.Image(plot_paths['class_metrics']),
        }
        
        if 'regression' in plot_paths:
            plots[f"plots/{prefix}_regression_scatter"] = wandb.Image(plot_paths['regression'])
        
        if 'damage_distribution_matrix' in plot_paths:
            plots[f"plots/{prefix}_damage_distribution_matrix"] = wandb.Image(plot_paths['damage_distribution_matrix'])
        
        wandb.log(plots, step=global_step)
    
    return plot_paths