import os
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import argparse

# Import the _process_single_file function from dataset.py
from dataset import DamageOutcomeDataset

def extract_positive_samples(data_dir, output_file):
    """Extract all positive samples from the dataset and save to a file.
    
    Args:
        data_dir: Directory containing the dataset files
        output_file: Path to save the positive samples
    """
    # Get list of all files
    files = os.listdir(data_dir)
    print(f"Found {len(files)} files")
    
    # Use multiprocessing to process files in parallel
    file_paths = [os.path.join(data_dir, file) for file in files]
    
    # Determine number of processes to use (use 75% of available CPUs)
    num_processes = max(1, int(cpu_count() * 0.9))
    print(f"Using {num_processes} processes for parallel processing")
    
    # Use multiprocessing Pool to process files in parallel
    positive_samples = []
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(DamageOutcomeDataset._process_single_file, file_paths),
            total=len(file_paths),
            desc="Processing files"
        ))
    
    # Extract positive samples
    for file_data in results:
        positive_file_samples = [item for item in file_data if item[13]]  # item[13] is has_damage
        positive_samples.extend(positive_file_samples)
    
    print(f"Found {len(positive_samples)} positive samples")
    
    # Save to file
    np.save(output_file, positive_samples)
    print(f"Saved positive samples to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract positive samples from dataset')
    parser.add_argument('--data_dir', type=str, default='../data/player_seq_allmap_full_npz_damage',
                        help='Directory containing dataset files')
    parser.add_argument('--output_file', type=str, default='../data/positive_samples.npy',
                        help='Output file to save positive samples')
    
    args = parser.parse_args()
    
    extract_positive_samples(args.data_dir, args.output_file)