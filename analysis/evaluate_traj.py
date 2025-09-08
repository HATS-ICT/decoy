import numpy as np
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
import pandas as pd
from functools import partial
from utils import transform_csgo_to_panda3d

def process_trajectory(args):
    """Process a single trajectory for all players and metrics"""
    from utils import transform_csgo_to_panda3d
    from evaluation import (procrustes_disparity, dtw_distance_normalized, 
                          euclidean_distance_normalized, rmse, frechet_distance,
                          interpolate_trajectory)
    
    traj_id, player_trajectory, waypoint_trajectory = args
    
    try:
        # Add variable to track max trajectory length
        max_trajectory_length = 0
        
        metrics = [
            (procrustes_disparity, {'xy_only': False}),  
            (procrustes_disparity, {'xy_only': True}),  
            (dtw_distance_normalized, {'xy_only': False}),  
            (dtw_distance_normalized, {'xy_only': True}),  
            (euclidean_distance_normalized, {'xy_only': False}),
            (euclidean_distance_normalized, {'xy_only': True}),
            (rmse, {'xy_only': False}),
            (rmse, {'xy_only': True}),
            (frechet_distance, {'xy_only': True})
        ]
        
        results = []
        
        player_traj = player_trajectory["player_trajectory"]
        waypoint_traj = waypoint_trajectory["player_trajectory"]
        player_traj_length = player_trajectory["player_seq_len"]
        waypoint_traj_length = waypoint_trajectory["player_seq_len"]

        for player_idx in range(10):
            player_traj_len = player_traj_length[player_idx]
            waypoint_traj_len = waypoint_traj_length[player_idx]
            player_traj_sample = player_traj[player_idx][:player_traj_len]
            waypoint_traj_sample = waypoint_traj[player_idx][:waypoint_traj_len]

            # Update max trajectory length
            current_max = max(player_traj_len, waypoint_traj_len)
            max_trajectory_length = max(max_trajectory_length, current_max)

            # Skip if either trajectory is empty or contains invalid values
            if (len(player_traj_sample) == 0 or len(waypoint_traj_sample) == 0 or
                np.any(np.isnan(player_traj_sample)) or np.any(np.isnan(waypoint_traj_sample)) or
                np.any(np.isinf(player_traj_sample)) or np.any(np.isinf(waypoint_traj_sample))):
                print(f"Warning: Invalid trajectory data found in {traj_id} for player {player_idx}")
                player_results = [np.nan] * (len(metrics) + 2)  # +2 for both trajectory lengths
                results.append(player_results)
                continue

            interpolated_len = 451
            player_traj_interp = interpolate_trajectory(player_traj_sample, interpolated_len, method='linear')
            waypoint_traj_interp = interpolate_trajectory(waypoint_traj_sample, interpolated_len, method='linear')

            player_results = []
            for metric_fn, metric_fn_kwargs in metrics:
                score = metric_fn(player_traj_interp, waypoint_traj_interp, **metric_fn_kwargs)
                player_results.append(score)
            # Add trajectory length as an additional metric
            player_results.append(player_traj_len)
            player_results.append(waypoint_traj_len)
            results.append(player_results)
        
        return traj_id, results, max_trajectory_length
    except Exception as e:
        print(f"Error processing trajectory {traj_id}: {str(e)}")
        return traj_id, [[np.nan] * (len(metrics) + 2)] * 10, 0  # Return max_length as 0 for failed cases

def load_trajectories(max_trajectories=None, do_transform=True):
    """Load trajectory data from files"""
    player_data_path = Path('data/player_seq_allmap_de_dust2_npz')
    waypoint_data_path = Path('logs')

    player_trajectories = {}
    waypoint_trajectories = {}
    
    player_files = list(player_data_path.glob('*.npz'))
    waypoint_files = list(waypoint_data_path.glob('*.npz'))

    player_stems = {f.stem for f in player_files}
    waypoint_stems = {f.stem for f in waypoint_files}
    common_stems = player_stems.intersection(waypoint_stems)
    
    player_files = [f for f in player_files if f.stem in common_stems]
    waypoint_files = [f for f in waypoint_files if f.stem in common_stems]

    if max_trajectories is not None:
        player_files = player_files[:max_trajectories]
        waypoint_files = waypoint_files[:max_trajectories]

    for player_file in player_files:
        npz_data = np.load(player_file, allow_pickle=True)
        player_trajectories[player_file.stem] = {key: npz_data[key] for key in npz_data.files}

    for waypoint_file in waypoint_files:
        npz_data = np.load(waypoint_file, allow_pickle=True)
        waypoint_trajectories[waypoint_file.stem] = {key: npz_data[key] for key in npz_data.files}

    if do_transform:
        for key, trajectory in player_trajectories.items():
            trajectory['player_trajectory'] = transform_csgo_to_panda3d(trajectory['player_trajectory'])
    
    return player_trajectories, waypoint_trajectories

def aggregate_player_metrics(data: np.ndarray, metric_names: list):
    """Aggregates player metrics with mean, std, min, and max for each team and overall"""
    n, m = data.shape
    if n % 10 != 0:
        raise ValueError("Number of players (n) must be a multiple of 10")
    
    df = pd.DataFrame(data, columns=metric_names)
    df["Team"] = [1 if i % 10 < 5 else 2 for i in range(n)]
    
    t_stats = df[df["Team"] == 1].drop(columns=["Team"]).agg(['mean', 'std', 'min', 'max']).T
    t_stats.columns = ['T_Mean', 'T_Std', 'T_Min', 'T_Max']
    
    ct_stats = df[df["Team"] == 2].drop(columns=["Team"]).agg(['mean', 'std', 'min', 'max']).T
    ct_stats.columns = ['CT_Mean', 'CT_Std', 'CT_Min', 'CT_Max']
    
    overall_stats = df.drop(columns=["Team"]).agg(['mean', 'std', 'min', 'max']).T
    overall_stats.columns = ['Overall_Mean', 'Overall_Std', 'Overall_Min', 'Overall_Max']
    
    return pd.concat([t_stats, ct_stats, overall_stats], axis=1)

def main():
    # Load trajectories
    player_trajectories, waypoint_trajectories = load_trajectories()
    common_traj_ids = list(set(player_trajectories.keys()) & set(waypoint_trajectories.keys()))
    
    # Prepare arguments for parallel processing
    process_args = [(traj_id, player_trajectories[traj_id], waypoint_trajectories[traj_id]) 
                   for traj_id in common_traj_ids]
    
    # Set up multiprocessing
    num_processes = mp.cpu_count()
    print(f"Using {num_processes} processes")
    
    # Run parallel processing with progress bar
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_trajectory, process_args), 
                          total=len(process_args),
                          desc="Processing trajectories"))
    
    # Separate valid results and failed trajectory IDs
    valid_results = []
    overall_max_length = 0
    for traj_id, result, max_length in results:
        if any(not np.all(np.isnan(player_result)) for player_result in result):
            valid_results.extend(result)
            overall_max_length = max(overall_max_length, max_length)
        else:
            print(f"Skipping trajectory {traj_id} due to invalid data")
    
    print(f"Maximum trajectory length across all valid trajectories: {overall_max_length}")
    
    evaluation_results_np = np.array(valid_results)
    
    # Define metric names
    metric_names = [
        "Procrustes",
        "Procrustes 2D",
        "DTW",
        "DTW 2D",
        "Euclidean",
        "Euclidean 2D",
        "RMSE",
        "RMSE 2D",
        "Frechet 2D",
        "Player_Trajectory_Length",
        "Waypoint_Trajectory_Length"
    ]
    
    # Aggregate and save results
    evaluation_result_df = aggregate_player_metrics(evaluation_results_np, metric_names)
    evaluation_result_df.to_csv("trajectory_metrics_parallel_v2.csv")
    print("Results saved to trajectory_metrics_parallel_v2.csv")

if __name__ == "__main__":
    main()