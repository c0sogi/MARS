import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from library.config import Config
from library.utils import load_dataset, compute_kinematics


class GestureDataset(Dataset):
    """
    PyTorch Dataset for the Robust Gated High-Capacity Monotonic Network.

    Implements:
    1. Loading of multi-modal data (Skeleton + Audio).
    2. Sliding window generation.
    3. Kinematically Consistent Augmentation (Rotate/Scale Pos -> Compute Vel/Acc).
    4. Feature Fusion (Pos + Vel + Acc + Audio).
    """

    def __init__(self, metadata_path, cache_path, is_train=True, load_cached_data=True):
        """
        Args:
            metadata_path (str): Path to the CSV metadata file.
            cache_path (str): Path to the .npz cache file.
            is_train (bool): If True, applies data augmentation.
            load_cached_data (bool): If True, attempts to load from cache first.
        """
        self.is_train = is_train

        # Load data using the library utility
        # This handles the heavy lifting of parsing .mat files and caching raw sequences
        raw_data = load_dataset(
            metadata_path, cache_path, load_cached_data=load_cached_data
        )

        self.skeletons = raw_data["skeletons"]
        self.audio = raw_data["audio"]
        self.labels = raw_data["labels"]
        self.sample_ids = raw_data["sample_ids"]

        # Generate Sliding Windows
        self.windows = []
        self.window_size = Config.WINDOW_SIZE
        self.stride = Config.STRIDE_TRAIN if is_train else Config.STRIDE_TEST

        for idx, (skel, aud, lbl) in enumerate(
            zip(self.skeletons, self.audio, self.labels)
        ):
            num_frames = skel.shape[0]

            # If sequence is shorter than window, we take one window with padding
            if num_frames < self.window_size:
                self.windows.append(
                    {
                        "sample_idx": idx,
                        "start": 0,
                        "end": num_frames,
                        "pad": self.window_size - num_frames,
                    }
                )
            else:
                # Generate sliding windows
                for start in range(0, num_frames - self.window_size + 1, self.stride):
                    self.windows.append(
                        {
                            "sample_idx": idx,
                            "start": start,
                            "end": start + self.window_size,
                            "pad": 0,
                        }
                    )

                # Handle the tail of the sequence if testing (to ensure full coverage)
                # For training, we skip the partial tail to avoid bias or excessive padding
                if not is_train:
                    last_start = self.windows[-1]["start"]
                    if last_start + self.window_size < num_frames:
                        # Add a final window aligned to the end
                        self.windows.append(
                            {
                                "sample_idx": idx,
                                "start": num_frames - self.window_size,
                                "end": num_frames,
                                "pad": 0,
                            }
                        )

    def __len__(self):
        return len(self.windows)

    def _augment_skeleton(self, positions):
        """
        Applies random rotation (Y-axis) and scaling to skeleton positions.
        positions: (T, J, 3)
        """
        # Random Scale: 0.9 to 1.1
        scale = np.random.uniform(0.9, 1.1)
        positions = positions * scale

        # Random Rotation around Y-axis: -15 to +15 degrees
        angle_deg = np.random.uniform(-15, 15)
        angle_rad = np.deg2rad(angle_deg)

        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # Rotation matrix for Y-axis
        # [ cos  0  sin]
        # [  0   1   0 ]
        # [-sin  0  cos]

        x = positions[..., 0]
        z = positions[..., 2]

        new_x = x * cos_a + z * sin_a
        new_z = -x * sin_a + z * cos_a

        positions[..., 0] = new_x
        positions[..., 2] = new_z

        return positions

    def __getitem__(self, idx):
        window_info = self.windows[idx]
        sample_idx = window_info["sample_idx"]
        start = window_info["start"]
        end = window_info["end"]
        pad = window_info["pad"]

        # 1. Extract Raw Data
        skel_window = self.skeletons[sample_idx][start:end].copy()  # (T, 20, 3)
        audio_window = self.audio[sample_idx][start:end].copy()  # (T, 13)
        label_window = self.labels[sample_idx][start:end].copy()  # (T,)

        # 2. Pad if necessary (Zero padding)
        if pad > 0:
            # Pad Skeleton: Repeat last frame or Zero? Zero is safer for kinematics (no movement)
            # but repeating last frame implies "static".
            # Using zero padding for features is standard.
            skel_pad = np.zeros((pad, 20, 3), dtype=np.float32)
            audio_pad = np.zeros((pad, 13), dtype=np.float32)
            label_pad = np.zeros((pad,), dtype=np.int32)  # 0 is background

            skel_window = np.concatenate([skel_window, skel_pad], axis=0)
            audio_window = np.concatenate([audio_window, audio_pad], axis=0)
            label_window = np.concatenate([label_window, label_pad], axis=0)

        # 3. Augmentation (Training Only)
        if self.is_train:
            skel_window = self._augment_skeleton(skel_window)

        # 4. Compute Kinematics (Pos -> Vel -> Acc)
        # Input: (T, 20, 3) -> Output: (T, 20, 9)
        # This ensures V and A are consistent with the (potentially augmented) P
        kinematics = compute_kinematics(skel_window)

        # 5. Flatten Skeleton Features
        # (T, 20, 9) -> (T, 180)
        T, J, C = kinematics.shape
        skel_flat = kinematics.reshape(T, J * C)

        # 6. Early Fusion (Concatenate Audio)
        # (T, 180) + (T, 13) -> (T, 193)
        features = np.concatenate([skel_flat, audio_window], axis=1)

        # 7. Convert to Tensors
        features_tensor = torch.from_numpy(features).float()
        labels_tensor = torch.from_numpy(label_window).long()

        # Return metadata for inference reconstruction if needed
        return features_tensor, labels_tensor, sample_idx, start


def get_dataloaders():
    """
    Factory function to create train, validation, and test dataloaders.
    """
    # Train Set
    train_dataset = GestureDataset(
        metadata_path=Config.TRAIN_METADATA_PATH,
        cache_path=Config.TRAIN_CACHE_PATH,
        is_train=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Validation Set
    val_dataset = GestureDataset(
        metadata_path=Config.VAL_METADATA_PATH,
        cache_path=Config.VAL_CACHE_PATH,
        is_train=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Test Set
    test_dataset = GestureDataset(
        metadata_path=Config.TEST_METADATA_PATH,
        cache_path=Config.TEST_CACHE_PATH,
        is_train=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
