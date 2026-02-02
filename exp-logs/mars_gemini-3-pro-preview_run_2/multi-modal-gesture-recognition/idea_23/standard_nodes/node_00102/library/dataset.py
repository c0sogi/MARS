import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.features import prepare_datasets
from library.utils import collate_fn


class GestureDataset(Dataset):
    """
    PyTorch Dataset for the Gesture Recognition task.
    Handles loading of cached features, applying physically consistent augmentations,
    and formatting data for the model.
    """

    def __init__(self, split, augment=False, load_cached_data=True):
        """
        Args:
            split (str): One of 'train', 'val', 'test'.
            augment (bool): Whether to apply data augmentation (smooth noise).
            load_cached_data (bool): Whether to try loading from disk cache.
        """
        self.split = split
        self.augment = augment

        # Load data using the provided library function which handles caching logic
        # prepare_datasets returns a dict with keys 'train', 'val', 'test'
        all_data = prepare_datasets(load_cached_data=load_cached_data)

        if split not in all_data:
            raise ValueError(f"Split '{split}' not found in prepared datasets.")

        data = all_data[split]

        # Data is stored as lists of numpy arrays
        self.features = data["features"]
        self.labels = data["labels"]
        self.boundaries = data["boundaries"]
        self.sample_ids = data["sample_ids"]

    def __len__(self):
        return len(self.features)

    def _apply_augmentation(self, features):
        """
        Applies physically consistent smooth noise to skeleton joints.
        1. Adds temporally smoothed Gaussian noise to joint positions.
        2. Re-computes Velocities based on noisy positions.
        3. Re-computes Bone Vectors based on noisy positions.
        4. Re-assembles the feature vector.

        Args:
            features (np.array): Shape (T, 118).

        Returns:
            np.array: Augmented features of shape (T, 118).
        """
        T_len = features.shape[0]

        # --- Feature Indices ---
        # Pos: 0-35 (12 joints * 3)
        # Vel: 36-71 (12 joints * 3)
        # Bones: 72-104 (11 bones * 3)
        # Audio: 105-117 (13 MFCCs)

        # 1. Extract Positions
        pos_dim = 36
        pos_flat = features[:, :pos_dim]
        pos = pos_flat.reshape(T_len, 12, 3)

        # 2. Generate Noise
        # Sigma = 5mm (0.005m) is a reasonable noise level for Kinect data
        sigma = 0.005
        noise = np.random.normal(0, sigma, size=pos.shape)

        # 3. Smooth Noise Temporally
        # Use a simple [0.25, 0.5, 0.25] kernel to simulate temporal correlation
        kernel = np.array([0.25, 0.5, 0.25])
        noise_smooth = np.zeros_like(noise)

        # Apply convolution along time axis for each coordinate of each joint
        for j in range(12):
            for c in range(3):
                noise_smooth[:, j, c] = np.convolve(noise[:, j, c], kernel, mode="same")

        # Add noise to positions
        new_pos = pos + noise_smooth

        # 4. Recompute Velocities
        # v[t] = p[t] - p[t-1]
        new_vel = np.zeros_like(new_pos)
        new_vel[1:] = new_pos[1:] - new_pos[:-1]

        # 5. Recompute Bone Vectors
        # Config.BONE_PAIRS contains indices (0-11) relative to the selected joints
        bones_list = []
        for p_idx, c_idx in Config.BONE_PAIRS:
            # Vector from Parent to Child
            bone_vec = new_pos[:, c_idx, :] - new_pos[:, p_idx, :]
            bones_list.append(bone_vec)

        new_bones = np.stack(bones_list, axis=1)  # Shape (T, 11, 3)

        # 6. Reassemble Feature Vector
        new_pos_flat = new_pos.reshape(T_len, -1)
        new_vel_flat = new_vel.reshape(T_len, -1)
        new_bones_flat = new_bones.reshape(T_len, -1)

        # Audio features remain unchanged
        audio_start_idx = 105
        audio_feats = features[:, audio_start_idx:]

        augmented_features = np.concatenate(
            [new_pos_flat, new_vel_flat, new_bones_flat, audio_feats], axis=1
        )

        return augmented_features.astype(np.float32)

    def __getitem__(self, idx):
        # Retrieve data
        feat = self.features[idx]  # (T, 118)
        lbl = self.labels[idx]  # (T,)
        bnd = self.boundaries[idx]  # (T,)
        sid = self.sample_ids[idx]

        # Apply augmentation if enabled (usually for training)
        if self.augment:
            feat = self._apply_augmentation(feat)

        # Convert to PyTorch tensors
        feat_tensor = torch.from_numpy(feat).float()
        lbl_tensor = torch.from_numpy(lbl).long()
        bnd_tensor = torch.from_numpy(bnd).float()

        # Return tuple compatible with collate_fn
        return feat_tensor, lbl_tensor, bnd_tensor, sid


def get_dataloader(
    split, batch_size=Config.BATCH_SIZE, shuffle=True, augment=False, num_workers=2
):
    """
    Creates a DataLoader for the specified split.

    Args:
        split (str): 'train', 'val', or 'test'.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the data.
        augment (bool): Whether to apply augmentation.
        num_workers (int): Number of worker processes.

    Returns:
        DataLoader: Configured PyTorch DataLoader.
    """
    dataset = GestureDataset(split, augment=augment)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=(split == "train"),  # Drop last incomplete batch only for training
    )

    return loader
