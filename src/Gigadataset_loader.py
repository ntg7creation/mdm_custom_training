import os
from os.path import join as pjoin
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class GigaHandsT2M(Dataset):
    """
    Core Dataset Loader for GigaHands T2M format.

    This class reads annotations from a JSONL file, finds the corresponding motion .npy files
    (either left or right hand), normalizes them using provided mean and std values, and returns
    them in the shape expected by the MDM model.

    Output:
        Each sample is a dictionary with:
            - 'inp': normalized motion tensor [263, 1, T]
            - 'text': corresponding textual annotation
            - 'lengths': number of frames
            - 'key': file path for reference

    This class is intended to be used internally by a wrapper that conforms to MDM's dataset expectations.
    """
    def __init__(self, root_dir, annotation_file, mean_std_dir, side='left', split='train', device='cpu',num_frames=120):
        assert side in ['left', 'right']
        self.side = side
        self.root_dir = root_dir
        self.device = device
        self.num_frames = num_frames

        self.mean = np.load(pjoin(mean_std_dir, f'mean_{side}.npy'))
        self.std = np.load(pjoin(mean_std_dir, f'std_{side}.npy'))

        self.mean_gpu = torch.tensor(self.mean).to(device)[None, :, None, None]
        self.std_gpu = torch.tensor(self.std).to(device)[None, :, None, None]

        self.samples = []  # list of (motion_path, text)
        self._load_annotations(annotation_file, split)

    def _load_annotations(self, annotation_file, split):
        with open(annotation_file, 'r') as f:
            lines = f.readlines()

        for line in tqdm(lines, desc=f"Loading GigaHands [{split}]"):
            ann = json.loads(line)
            scene = ann['scene']
            seq = ann['sequence']
            text_list = ann['rewritten_annotation']
            text = text_list[0]

            motion_path = pjoin(self.root_dir, scene, 'keypoints_3d', seq, f'dmvb_{self.side}.npy')
            if os.path.exists(motion_path):
                self.samples.append((motion_path, text))

        assert len(self.samples) > 0, "No valid samples found."

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        motion_path, text = self.samples[idx]
        motion = np.load(motion_path).astype(np.float32)  # [T, 263]
        epsilon = 1e-8
        motion = (motion - self.mean) / (self.std + epsilon)

        # Crop or pad to fixed length if needed
        if self.fixed_len > 0:
            T = motion.shape[0]
            if T >= self.fixed_len:
                start = np.random.randint(0, T - self.fixed_len + 1)
                motion = motion[:self.num_frames]
            else:
                pad = np.zeros((self.fixed_len - T, motion.shape[1]), dtype=np.float32)
                motion = np.concatenate([motion, pad], axis=0)  # pad at end

        motion = torch.tensor(motion.T).unsqueeze(1)  # [263, 1, T]

        return {
            'inp': motion,
            'text': text,
            'lengths': motion.shape[-1],
            'key': motion_path
        }


class GigaHandsML3D(Dataset):
    """
    Wrapper class to conform to MDM's dataset interface.

    The original MDM expects datasets like HumanML3D or KIT to have:
        - .mean, .std, .mean_gpu, .std_gpu
        - .t2m_dataset attribute (an instance of the real loader)
        - .__getitem__ and .__len__

    This wrapper initializes a GigaHandsT2M instance and stores it in self.t2m_dataset,
    forwarding calls and exposing normalized statistics.

    Use this as the value for --dataset in MDM training CLI (after registration).

    Example:
        python train_mdm.py --dataset gigahands ...
    """
    def __init__(self, mode, annotation_file, root_dir, mean_std_dir, side='left', split='train', **kwargs):
        self.mode = mode
        self.dataset_name = 'gigahands'
        self.dataname = 'gigahands'

        abs_base_path = kwargs.get('abs_path', '.')
        device = kwargs.get('device', None)

        self.mean = np.load(pjoin(mean_std_dir, f'mean_{side}.npy'))
        self.std = np.load(pjoin(mean_std_dir, f'std_{side}.npy'))

        self.split_file = None  # Not needed since we use annotations.jsonl

        self.t2m_dataset = GigaHandsT2M(
            root_dir=pjoin(abs_base_path, root_dir),
            annotation_file=pjoin(abs_base_path, annotation_file),
            mean_std_dir=pjoin(abs_base_path, mean_std_dir),
            side=side,
            num_frames=kwargs.get('num_frames', 120),
            split=split,
            device=device or 'cpu'
        )

        self.mean_gpu = torch.tensor(self.mean).to(device)[None, :, None, None]
        self.std_gpu = torch.tensor(self.std).to(device)[None, :, None, None]

        self.num_actions = 1  # Dummy for MDM
        assert len(self.t2m_dataset) > 1, 'You loaded an empty dataset.'

    def __getitem__(self, item):
        return self.t2m_dataset[item]

    def __len__(self):
        return len(self.t2m_dataset)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', required=True)
    parser.add_argument('--annotation_file', required=True)
    parser.add_argument('--mean_std_dir', required=True)
    parser.add_argument('--side', choices=['left', 'right'], default='left')
    parser.add_argument('--split', default='train')
    args = parser.parse_args()

    dataset = GigaHandsT2M(
        root_dir=args.root_dir,
        annotation_file=args.annotation_file,
        mean_std_dir=args.mean_std_dir,
        side=args.side,
        split=args.split
    )

    print(f"Loaded {len(dataset)} samples")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"[{i}] {sample['key']}: {sample['text']} → motion shape {sample['inp'].shape}")
