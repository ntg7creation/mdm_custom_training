# mdm_custom_training/src/convert_gigahands_to_mdm.py

import os
import json
import numpy as np
from tqdm import tqdm

# Constants for MDM format
NUM_JOINTS = 22
RIC_DIM = (NUM_JOINTS - 1) * 3
ROT6D_DIM = (NUM_JOINTS - 1) * 6
VEL_DIM = NUM_JOINTS * 3
TOTAL_DIM = 1 + 2 + 1 + RIC_DIM + ROT6D_DIM + VEL_DIM + 4  # = 263
FPS = 20  # Default HumanML3D frame rate

def load_jsonl(filepath):
    with open(filepath, "r") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def strip_w(pts):
    return np.array([[x, y, z] for x, y, z, _ in pts], dtype=np.float32)

def compute_velocity(positions):
    return positions[1:] - positions[:-1]  # [T-1, J, 3]

def compute_root_features(positions):
    root_pos = positions[:, 0]  # root is at index 0
    delta = root_pos[1:] - root_pos[:-1]  # [T-1, 3]
    rot_vel = np.arctan2(delta[:, 0], delta[:, 2]).reshape(-1, 1)  # Y-rotation only
    lin_vel = delta[:, [0, 2]]  # XZ
    root_y = root_pos[:-1, 1:2]  # Y
    root_feats = np.concatenate([rot_vel, lin_vel, root_y], axis=1)
    # print("Root features shape:", root_feats.shape)
    return root_feats  # [T-1, 4]

def convert_sequence(single_hand):
    T = len(single_hand)
    positions = np.array(single_hand[:T])  # [T, 22, 3] after root duplication

    # Compute features
    root_feats = compute_root_features(positions)  # [T-1, 4]

    ric = positions[1:, 1:] - positions[1:, [0]]  # relative to original root [T-1, 21, 3]
    ric = ric.reshape(len(ric), -1)  # [T-1, 63]
    # print("RIC shape:", ric.shape)

    rot6d = np.zeros((len(ric), 126), dtype=np.float32)  # [T-1, 126]
    # print("ROT6D shape:", rot6d.shape)

    velocities = compute_velocity(positions)  # [T-1, 22, 3]
    vel = velocities.reshape(len(velocities), -1)  # [T-1, 66]
    # print("Velocity shape:", vel.shape)

    contact = np.zeros((len(ric), 4), dtype=np.float32)  # [T-1, 4]
    # print("Contact shape:", contact.shape)

    full_dmvb = np.concatenate([root_feats, ric, rot6d, vel, contact], axis=1)  # [T-1, 263]
    # print("Final DMVB shape:", full_dmvb.shape)
    return full_dmvb


def main():
    base_path = os.path.join(os.path.dirname(__file__), "..", "GigaHands")
    annotation_path = os.path.join(base_path, "annotations_v2.jsonl")
    annotations = load_jsonl(annotation_path)

    for ann in tqdm(annotations):
        seq_name = ann["sequence"]
        scene = ann["scene"]

        motion_dir = os.path.join(base_path, "hand_poses", scene, "keypoints_3d", seq_name)
        output_dir = os.path.join(os.path.dirname(__file__), "..", "converted_motions", "hand_poses", scene, "keypoints_3d", seq_name)
        os.makedirs(output_dir, exist_ok=True)

        try:
            left = load_jsonl(os.path.join(motion_dir, "left.jsonl"))
            right = load_jsonl(os.path.join(motion_dir, "right.jsonl"))

            def duplicate_root(frames):
                return [np.concatenate([f[[0]], f], axis=0) for f in frames]  # insert duplicate of root at index 0

            left = [strip_w(f) for f in left]
            right = [strip_w(f) for f in right]

            left = duplicate_root(left)
            right = duplicate_root(right)

            motion_left = convert_sequence(left)
            motion_right = convert_sequence(right)

            np.save(os.path.join(output_dir, "dmvb_left.npy"), motion_left)
            np.save(os.path.join(output_dir, "dmvb_right.npy"), motion_right)
        except Exception as e:
            print(f"Error processing {scene}/{seq_name}: {e}")

if __name__ == "__main__":
    main()

