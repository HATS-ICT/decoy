import subprocess
import itertools
import os
import time
from datetime import datetime

def main():
    # Define hyperparameter values to sweep
    weighted_cross_entropy_options = [True, False]
    weighted_mse_options = [True, False]
    model_modes = ["joint", "causal", "causal_embed"]
    
    # Create all combinations
    all_combinations = list(itertools.product(weighted_cross_entropy_options, weighted_mse_options, model_modes))
    print(f"Total combinations to try: {len(all_combinations)}")
    
    # Number of GPUs and processes per GPU
    num_gpus = 8
    processes_per_gpu = 1
    max_processes = num_gpus * processes_per_gpu
    
    # Create a directory for logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"hyperparam_sweep_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Track running processes
    running_processes = []
    completed_count = 0
    
    # Process all combinations
    for i, (weighted_ce, weighted_mse, model_mode) in enumerate(all_combinations):
        # Determine which GPU to use for this run
        gpu_id = (i % max_processes) // processes_per_gpu
        
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
        
        # Prepare command
        run_id = f"run_{i+1:03d}_wce{int(weighted_ce)}_wmse{int(weighted_mse)}_mode{model_mode}"
        log_file = os.path.join(log_dir, f"{run_id}.log")
        
        # Build command as a list
        cmd = [
            "python", "main.py", 
            "--weighted_cross_entropy", str(weighted_ce).lower(),
            "--weighted_mse", str(weighted_mse).lower(),
            "--model_mode", model_mode
        ]
        
        # Add a unique seed for each run
        cmd.extend(["--seed", str(1000 + i)])
        
        # Set environment variables
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # Start the process
        print(f"Starting {run_id} on GPU {gpu_id}: {' '.join(cmd)} (CUDA_VISIBLE_DEVICES={gpu_id})")
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
                'params': {
                    'weighted_cross_entropy': weighted_ce,
                    'weighted_mse': weighted_mse,
                    'model_mode': model_mode
                }
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

if __name__ == "__main__":
    main()