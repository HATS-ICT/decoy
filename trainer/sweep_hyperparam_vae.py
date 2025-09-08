import subprocess
import random
import os
import time
from datetime import datetime
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run hyperparameter sweep for VAE model')
    parser.add_argument('--num_runs', type=int, default=20, 
                        help='Number of random hyperparameter combinations to try')
    parser.add_argument('--num_gpus', type=int, default=8, 
                        help='Number of GPUs available')
    parser.add_argument('--processes_per_gpu', type=int, default=1, 
                        help='Number of processes to run per GPU')
    args = parser.parse_args()
    
    # Define hyperparameter values to sweep
    hyperparams = {
        'hidden_dim': [64, 128, 256, 512],
        'learning_rate': [3e-4, 0.001],
        'latent_dim': [8, 16, 32],
        'batch_size': [32, 64, 128, 256],
        'dropout': [0, 0.1, 0.3],
        'use_xavier_init': [True, False],
        'use_batchnorm': [True, False],
        'use_gradient_clipping': [True, False],
        'kl_weight': [0.1, 0.5, 1, 2]
    }
    
    # Calculate total possible combinations
    total_combinations = 1
    for values in hyperparams.values():
        total_combinations *= len(values)
    
    print(f"Total possible combinations: {total_combinations}")
    print(f"Running {args.num_runs} random combinations")
    
    # Number of GPUs and processes per GPU
    max_processes = args.num_gpus * args.processes_per_gpu
    
    # Create a directory for logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"vae_hyperparam_sweep_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Save configuration details
    with open(os.path.join(log_dir, "sweep_config.txt"), 'w') as f:
        f.write(f"Hyperparameter sweep configuration:\n")
        f.write(f"Total possible combinations: {total_combinations}\n")
        f.write(f"Number of random combinations to try: {args.num_runs}\n")
        f.write(f"Number of GPUs: {args.num_gpus}\n")
        f.write(f"Processes per GPU: {args.processes_per_gpu}\n\n")
        f.write("Hyperparameter space:\n")
        for param, values in hyperparams.items():
            f.write(f"{param}: {values}\n")
    
    # Track running processes
    running_processes = []
    completed_count = 0
    
    # Track used combinations to avoid repetition
    used_combinations = set()
    
    # Process random combinations
    for i in range(args.num_runs):
        # Wait if we've reached the maximum number of concurrent processes
        while len(running_processes) >= max_processes:
            # Check if any processes have completed
            still_running = []
            for proc, proc_info in running_processes:
                if proc.poll() is None:  # Process is still running
                    still_running.append((proc, proc_info))
                else:
                    completed_count += 1
                    print(f"Completed run {proc_info['run_id']} with exit code {proc.returncode}")
            
            running_processes = still_running
            
            if len(running_processes) >= max_processes:
                time.sleep(10)  # Wait before checking again
        
        # Generate a random combination that hasn't been used yet
        while True:
            combo = tuple(random.choice(values) for values in hyperparams.values())
            if combo not in used_combinations:
                used_combinations.add(combo)
                break
        
        # Extract parameters
        hidden_dim, learning_rate, latent_dim, batch_size, dropout, use_xavier_init, \
        use_batchnorm, use_gradient_clipping, kl_weight = combo
        
        # Determine which GPU to use for this run
        gpu_id = (i % max_processes) // args.processes_per_gpu
        
        # Create a unique run ID
        run_id = f"run_{i+1:03d}_h{hidden_dim}_l{latent_dim}_lr{learning_rate}_b{batch_size}_d{dropout}_x{int(use_xavier_init)}_bn{int(use_batchnorm)}_gc{int(use_gradient_clipping)}_kl{kl_weight}"
        log_file = os.path.join(log_dir, f"{run_id}.log")
        
        # Build command as a list
        cmd = [
            "python", "regression_vae_model.py",
            "--hidden_dim", str(hidden_dim),
            "--learning_rate", str(learning_rate),
            "--latent_dim", str(latent_dim),
            "--batch_size", str(batch_size),
            "--dropout", str(dropout),
            "--use_xavier_init", str(use_xavier_init),
            "--use_batchnorm", str(use_batchnorm),
            "--use_gradient_clipping", str(use_gradient_clipping),
            "--kl_weight", str(kl_weight),
            "--num_epoch", "20",  # Fixed number of epochs for all runs
            "--use_wandb"  # Enable wandb logging for all runs
        ]
        
        # Add a unique seed for each run
        cmd.extend(["--seed", str(2000 + i)])
        
        # Set environment variables
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # Save the configuration to a separate file for easy reference
        config_file = os.path.join(log_dir, f"{run_id}_config.txt")
        with open(config_file, 'w') as f:
            f.write(f"Run ID: {run_id}\n")
            f.write(f"GPU ID: {gpu_id}\n")
            f.write(f"Command: {' '.join(cmd)}\n\n")
            f.write("Parameters:\n")
            param_names = list(hyperparams.keys())
            for j, param_name in enumerate(param_names):
                f.write(f"{param_name}: {combo[j]}\n")
        
        # Start the process
        print(f"Starting {run_id} on GPU {gpu_id} ({i+1}/{args.num_runs})")
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT
            )
        
        # Store process with its info
        running_processes.append((
            process, 
            {
                'run_id': run_id,
                'gpu_id': gpu_id,
                'params': dict(zip(hyperparams.keys(), combo))
            }
        ))
        
        # Small delay to prevent race conditions
        time.sleep(1)
    
    # Wait for all remaining processes to complete
    print("Waiting for all remaining processes to complete...")
    for proc, proc_info in running_processes:
        proc.wait()
        completed_count += 1
        print(f"Completed run {proc_info['run_id']} with exit code {proc.returncode}")
    
    print(f"All {completed_count} hyperparameter combinations completed!")
    print(f"Results and logs saved in: {log_dir}")

if __name__ == "__main__":
    main() 