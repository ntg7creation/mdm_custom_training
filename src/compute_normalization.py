import numpy as np
import os
from tqdm import tqdm

# One level up from src/, pointing to the dataset root
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'converted_motions', 'hand_poses'))

left_data = []
right_data = []

# Collect all dmvb paths first
dmvb_pairs = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    if 'dmvb_left.npy' in filenames and 'dmvb_right.npy' in filenames:
        dmvb_pairs.append((
            os.path.join(dirpath, 'dmvb_left.npy'),
            os.path.join(dirpath, 'dmvb_right.npy')
        ))

print(f"Found {len(dmvb_pairs)} motion pairs")

# Load them with progress bar
for left_path, right_path in tqdm(dmvb_pairs, desc="Loading motion vectors"):
    left = np.load(left_path)
    right = np.load(right_path)
    left_data.append(left)
    right_data.append(right)

# Concatenate and compute stats
left_data = np.concatenate(left_data, axis=0)
right_data = np.concatenate(right_data, axis=0)

mean_left = np.mean(left_data, axis=0)
std_left = np.std(left_data, axis=0)
mean_right = np.mean(right_data, axis=0)
std_right = np.std(right_data, axis=0)

# Save results
output_dir = os.path.join(root_dir, 'norm_stats')
os.makedirs(output_dir, exist_ok=True)
np.save(os.path.join(output_dir, 'mean_left.npy'), mean_left)
np.save(os.path.join(output_dir, 'std_left.npy'), std_left)
np.save(os.path.join(output_dir, 'mean_right.npy'), mean_right)
np.save(os.path.join(output_dir, 'std_right.npy'), std_right)

print(f"✅ Normalization stats saved to: {output_dir}")
