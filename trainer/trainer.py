import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import wandb
from dataset import collate_fn
from utils import evaluate_predictions, log_metrics, plot_metrics


class Trainer:
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

    def train(self, verbose: bool = True):
        while self.current_epoch < self.args.num_epoch:
            train_results = self.train_epoch()
            log_metrics(train_results['metrics'], self.current_epoch+1, self.global_step, self.run_dir, prefix='train', log_wandb=self.args.use_wandb)
            plot_metrics(train_results, self.current_epoch+1, self.global_step, self.run_dir, prefix='train', log_wandb=self.args.use_wandb)

            val_results = self.validate_epoch()
            log_metrics(val_results['metrics'], self.current_epoch+1, self.global_step, self.run_dir, prefix='val', log_wandb=self.args.use_wandb)
            plot_metrics(val_results, self.current_epoch+1, self.global_step, self.run_dir, prefix='val', log_wandb=self.args.use_wandb)

            self.current_epoch += 1
        
        if self.args.use_wandb:
            wandb.finish()
        return
    
    def train_epoch(self):
        return self._run_epoch(self.train_dataset, is_training=True)

    def validate_epoch(self):
        return self._run_epoch(self.val_dataset, is_training=False)
    
    def _process_batch(self, batch_x, batch_y, is_training=True):
        """Common processing for both training and validation batches"""
        batch_x = {k: v.to(self.device) for k, v in batch_x.items()}
        batch_y = {k: v.to(self.device) for k, v in batch_y.items()}
        
        if is_training:
            self.optimizer.zero_grad()
            
        outputs = self.model(batch_x)
        
        damage_mask = batch_y['damage_indicator'].squeeze() > 0
        total_loss, loss_items = self.total_loss(outputs, batch_y, damage_mask)
        
        if is_training:
            total_loss.backward()
            
            if self.args.use_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gradient_clipping)
                
            self.optimizer.step()
            
            self.global_step += 1
            if self.args.use_wandb and self.global_step % self.args.log_interval == 0:
                wandb.log({
                    'train_total_loss': loss_items['total_loss'],
                    'train_damage_indicator_loss': loss_items['damage_indicator_loss'],
                    'train_damage_value_loss': loss_items['damage_value_loss'],
                    'train_hit_group_loss': loss_items['hit_group_loss'],
                }, step=self.global_step)
        
        # Collect predictions and targets
        batch_size = batch_y['damage_indicator'].size(0)
        
        return {
            'batch_size': batch_size,
            'loss_items': loss_items,
            'total_loss': total_loss.item(),
            'predictions': {
                'damage_indicator_probs': outputs['damage_indicator_probs'].detach().cpu(),
                'damage_value': outputs['damage_value'].detach().cpu(),
                'hit_group_logits': outputs['hit_group_logits'].detach().cpu(),
            },
            'targets': {
                'damage_indicator': batch_y['damage_indicator'].detach().cpu(),
                'damage_value': batch_y['damage_value'].detach().cpu(),
                'hit_group': batch_y['hit_group'].detach().cpu(),
            }
        }
    
    def _run_epoch(self, dataset, is_training=True):
        """Common epoch processing for both training and validation"""
        mode = "Train" if is_training else "Val"
        if is_training:
            self.model.train()
        else:
            self.model.eval()
            
        dataset.load_next_chunk()
        
        data_loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=is_training,
            collate_fn=collate_fn
        )
        
        total_samples = 0
        losses = {
            'total_loss': 0.0,
            'damage_indicator_loss': 0.0,
            'damage_value_loss': 0.0,
            'hit_group_loss': 0.0
        }
        
        preds = {
            'damage_indicator_probs': [],
            'damage_value': [],
            'hit_group_logits': [],
        }
        
        targets = {
            'damage_indicator_targets': [],
            'damage_value_targets': [],
            'hit_group_targets': []
        }
        
        weapon_indices = []
        
        pbar = tqdm(data_loader, desc=f"Epoch {self.current_epoch+1}/{self.args.num_epoch} [{mode}]", total=len(data_loader))
        
        batch_process_fn = lambda x, y: self._process_batch(x, y, is_training)
        
        if not is_training:
            # Use torch.no_grad() for validation
            with torch.no_grad():
                for batch_idx, (batch_x, batch_y) in enumerate(pbar):
                    result = batch_process_fn(batch_x, batch_y)
                    self._update_metrics(result, losses, preds, targets, pbar, total_samples)
                    # Store weapon indices
                    weapon_indices.append(torch.argmax(batch_x['weapon'], dim=1).cpu())
                    total_samples += result['batch_size']
        else:
            for batch_idx, (batch_x, batch_y) in enumerate(pbar):
                result = batch_process_fn(batch_x, batch_y)
                self._update_metrics(result, losses, preds, targets, pbar, total_samples)
                # Store weapon indices
                weapon_indices.append(torch.argmax(batch_x['weapon'], dim=1).cpu())
                total_samples += result['batch_size']
        
        # Calculate average loss
        for k in losses:
            losses[k] /= total_samples
        
        # Concatenate all predictions and targets
        all_damage_indicator_probs = torch.cat(preds['damage_indicator_probs']).numpy()
        all_damage_value = torch.cat(preds['damage_value']).numpy()
        all_hit_group_logits = torch.cat(preds['hit_group_logits']).numpy()
        
        all_damage_indicator_targets = torch.cat(targets['damage_indicator_targets']).numpy()
        all_damage_value_targets = torch.cat(targets['damage_value_targets']).numpy()
        all_hit_group_targets = torch.cat(targets['hit_group_targets']).numpy()
        
        all_weapon_indices = torch.cat(weapon_indices).numpy()
        
        print(f"Evaluating {mode.lower()} predictions")
        metrics, predictions = evaluate_predictions(
            all_damage_indicator_probs,
            all_damage_value,
            all_hit_group_logits,
            all_damage_indicator_targets,
            all_damage_value_targets,
            all_hit_group_targets
        )
        metrics['loss'] = losses
        
        return {
            'metrics': metrics,
            'predictions': {
                'damage_indicator_probs': all_damage_indicator_probs,
                'damage_indicator_preds_default': predictions['damage_indicator_preds_default'],
                'damage_indicator_preds_optimal': predictions['damage_indicator_preds_optimal'],
                'damage_value': all_damage_value,
                'hit_group_logits': all_hit_group_logits,
                'hit_group_preds': predictions['hit_group_preds'],
                'damage_indicator_targets': all_damage_indicator_targets,
                'damage_value_targets': all_damage_value_targets,
                'hit_group_targets': all_hit_group_targets,
                'weapon_indices': all_weapon_indices
            }
        }
    
    def _update_metrics(self, result, losses, preds, targets, pbar, total_samples):
        """Update metrics with batch results"""
        batch_size = result['batch_size']
        loss_items = result['loss_items']
        
        # Update losses
        for k in losses:
            losses[k] += loss_items[k] * batch_size
        
        # Update progress bar
        pbar.set_postfix(loss=result['total_loss'])
        
        # Collect predictions and targets
        preds['damage_indicator_probs'].append(result['predictions']['damage_indicator_probs'])
        preds['damage_value'].append(result['predictions']['damage_value'])
        preds['hit_group_logits'].append(result['predictions']['hit_group_logits'])
        
        targets['damage_indicator_targets'].append(result['targets']['damage_indicator'])
        targets['damage_value_targets'].append(result['targets']['damage_value'])
        targets['hit_group_targets'].append(result['targets']['hit_group'])

    def total_loss(self, pred_dict, target_dict, damage_mask):
        def damage_indicator_loss_fn(pred, target):
            return F.binary_cross_entropy(pred, target)

        def damage_value_loss_fn(pred, target, mask, hit_group_targets=None):
            if not torch.any(mask):
                return torch.tensor(0.0, device=pred.device)
            
            if self.args.weighted_mse and hit_group_targets is not None:
                weights = self.train_dataset.hit_group_class_weights.to(pred.device)
                
                hit_groups = hit_group_targets[mask]
                squared_errors = (pred[mask] - target[mask]) ** 2
                return (weights[hit_groups] * squared_errors).mean()
            else:
                return F.mse_loss(pred[mask], target[mask])

        def hit_group_loss_fn(pred, target, mask):
            if not torch.any(mask):
                return torch.tensor(0.0, device=pred.device)
            
            if self.args.weighted_cross_entropy:
                weights = self.train_dataset.hit_group_class_weights.to(pred.device)
                return F.cross_entropy(pred[mask], target[mask], weight=weights)
            else:
                return F.cross_entropy(pred[mask], target[mask])
        
        damage_indicator_loss = damage_indicator_loss_fn(pred_dict['damage_indicator_probs'], target_dict['damage_indicator'])
        damage_value_loss = damage_value_loss_fn(pred_dict['damage_value'], target_dict['damage_value'], damage_mask, target_dict['hit_group'])
        hit_group_loss = hit_group_loss_fn(pred_dict['hit_group_logits'], target_dict['hit_group'], damage_mask)

        total_loss = self.args.damage_indicator_loss_weight * damage_indicator_loss + \
            self.args.damage_value_loss_weight * damage_value_loss + \
            self.args.hit_group_loss_weight * hit_group_loss

        loss_items = {
            'total_loss': total_loss.item(),
            'damage_indicator_loss': damage_indicator_loss.item(),
            'damage_value_loss': damage_value_loss.item(),
            'hit_group_loss': hit_group_loss.item()
        }
        return total_loss, loss_items

    def save_checkpoint(self, filename, is_best=False):
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        
        # Save regular checkpoint
        torch.save(checkpoint, os.path.join(self.run_dir, filename))
        
        # Save best model if this is the best so far
        if is_best:
            torch.save(self.model.state_dict(), os.path.join(self.run_dir, "best_model.pth"))
