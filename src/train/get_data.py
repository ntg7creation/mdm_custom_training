import platform
from torch.utils.data import DataLoader
from tensors import collate as default_collate  # Update this if your tensors.py is moved
from Gigadataset_loader import GigaHandsML3D  # Direct import since you’re not dynamically loading other datasets

NUM_WORKERS = 0 if platform.system() == 'Windows' else 8  # Adjust based on your system


def get_dataset(split='train', hml_mode='train', abs_path='.', fixed_len=0,
                device=None, autoregressive=False, cache_path=None, num_frames=None):
    dataset = GigaHandsML3D(
        mode=hml_mode,
        annotation_file='dataset/annotations_v2.jsonl',
        root_dir='converted_motions/hand_poses',
        mean_std_dir='converted_motions/hand_poses/norm_stats',
        side='left',  # or right — make this configurable if needed
        split=split,
        device=device,
        abs_path=abs_path
    )
    return dataset


def get_dataset_loader(batch_size, split='train', hml_mode='train', fixed_len=0, pred_len=0,
                       device=None, autoregressive=False, num_frames=None):
    dataset = get_dataset(split=split, hml_mode=hml_mode, fixed_len=fixed_len,
                          device=device, autoregressive=autoregressive, num_frames=num_frames)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True, collate_fn=default_collate
    )
    return loader
