import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from library.config import (
    INPUT_DIR,
    SEED,
    WINDOW_SIZE,
    STRIDE,
    NUM_JOINTS,
    JOINTS_DIM,
    N_MFCC,
    LABEL_MAP,
)
from library.data_utils import load_robust_mat, compute_kinematics, extract_audio_mfcc


class GestureDataset(Dataset):
    """
    Multi-modal Gesture Recognition Dataset.
    Implements Kinematically Consistent Augmentation and Sliding Window generation.
    """

    def __init__(self, metadata_file, is_train=True, load_cached_data=True, limit=None):
        """
        Args:
            metadata_file (str): Path to the metadata CSV file.
            is_train (bool): If True, applies data augmentation.
            load_cached_data (bool): If True, attempts to load pre-processed features from cache.
            limit (int, optional): Limit dataset size for debugging.
        """
        self.is_train = is_train
        self.window_size = WINDOW_SIZE
        self.stride = STRIDE

        # Ensure reproducibility
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        self.samples = []
        self.indices = []  # List of (sample_idx, start_frame) tuples

        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        df = pd.read_csv(metadata_file)
        if limit:
            df = df.head(limit)

        print(
            f"Initializing GestureDataset from {metadata_file} (is_train={is_train})..."
        )

        valid_samples = 0

        for _, row in df.iterrows():
            sample_id = row["sample_id"]
            data_path = os.path.join(INPUT_DIR, row["data_path"])
            audio_path = os.path.join(INPUT_DIR, row["audio_path"])

            # 1. Load Raw Skeleton Data (Positions)
            # Returns (T, J, 3) or None
            skeleton = load_robust_mat(data_path, load_cached_data=load_cached_data)

            if skeleton is None:
                continue

            num_frames = skeleton.shape[0]
            if num_frames == 0:
                continue

            # 2. Load Audio Data (MFCCs)
            # Returns (T, N_MFCC)
            audio = extract_audio_mfcc(
                audio_path, num_frames, load_cached_data=load_cached_data
            )

            # 3. Process Labels
            # Initialize with Background class (0)
            labels = np.zeros(num_frames, dtype=np.int64)

            if pd.notna(row["labels"]):
                try:
                    label_list = json.loads(row["labels"])
                    for l in label_list:
                        # Convert Matlab 1-based indexing to Python 0-based
                        # Assuming 'begin' is 1-based inclusive start
                        start = max(0, int(l["begin"]) - 1)
                        # Assuming 'end' is 1-based inclusive end
                        end = int(l["end"])
                        gid = int(l["id"])

                        if start < num_frames:
                            labels[start : min(end, num_frames)] = gid
                except (json.JSONDecodeError, ValueError):
                    pass  # Keep as background if parsing fails

            # Store raw data in memory
            self.samples.append(
                {
                    "sample_id": sample_id,
                    "skeleton": skeleton,  # (T, 20, 3)
                    "audio": audio,  # (T, 13)
                    "labels": labels,  # (T,)
                }
            )

            # 4. Generate Sliding Window Indices
            sample_idx = len(self.samples) - 1

            if num_frames <= self.window_size:
                # Sequence shorter than window: take one window (will be padded)
                self.indices.append((sample_idx, 0))
            else:
                # Sliding window
                current_start = 0
                while current_start + self.window_size <= num_frames:
                    self.indices.append((sample_idx, current_start))
                    current_start += self.stride

                # For validation/test, ensure we cover the end of the sequence
                if not self.is_train and current_start < num_frames:
                    # Add a final window aligned to the end
                    self.indices.append(
                        (sample_idx, max(0, num_frames - self.window_size))
                    )

            valid_samples += 1

        print(
            f"Loaded {valid_samples} valid samples. Generated {len(self.indices)} windows."
        )

    def __len__(self):
        return len(self.indices)

    def _augment(self, positions):
        """
        Applies random 3D rotation and scaling to skeleton positions.
        Args:
            positions (np.ndarray): Shape (T, J, 3)
        Returns:
            np.ndarray: Augmented positions (T, J, 3)
        """
        # 1. Random Scaling (0.9x to 1.1x)
        scale = np.random.uniform(0.9, 1.1)
        positions = positions * scale

        # 2. Random Rotation around Y-axis (+/- 30 degrees)
        angle_deg = np.random.uniform(-30, 30)
        angle_rad = np.deg2rad(angle_deg)

        c, s = np.cos(angle_rad), np.sin(angle_rad)
        # Rotation matrix for Y-axis
        R_y = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

        # Apply rotation: (T, J, 3) -> (T*J, 3) @ (3, 3) -> (T, J, 3)
        T, J, _ = positions.shape
        flat_pos = positions.reshape(-1, 3)
        rotated_flat = flat_pos @ R_y.T
        return rotated_flat.reshape(T, J, 3)

    def __getitem__(self, idx):
        sample_idx, start_frame = self.indices[idx]
        sample = self.samples[sample_idx]

        skel_full = sample["skeleton"]
        audio_full = sample["audio"]
        labels_full = sample["labels"]

        seq_len = skel_full.shape[0]

        # Determine window slice
        if seq_len < self.window_size:
            # Pad sequences shorter than window
            pad_len = self.window_size - seq_len

            # Skeleton: Repeat last frame
            skel_window = skel_full
            last_frame = skel_full[-1:]
            skel_pad = np.repeat(last_frame, pad_len, axis=0)
            skel_window = np.concatenate([skel_window, skel_pad], axis=0)

            # Audio: Pad with zeros
            audio_window = audio_full
            audio_pad = np.zeros((pad_len, N_MFCC), dtype=np.float32)
            audio_window = np.concatenate([audio_window, audio_pad], axis=0)

            # Labels: Pad with background (0)
            labels_window = labels_full
            labels_pad = np.zeros(pad_len, dtype=np.int64)
            labels_window = np.concatenate([labels_window, labels_pad], axis=0)

        else:
            end_frame = start_frame + self.window_size
            skel_window = skel_full[start_frame:end_frame]
            audio_window = audio_full[start_frame:end_frame]
            labels_window = labels_full[start_frame:end_frame]

        # --- Kinematically Consistent Augmentation ---
        # 1. Augment Raw Positions (Train only)
        if self.is_train:
            skel_window = self._augment(skel_window)

        # 2. Compute Derivatives (Velocity, Acceleration)
        # Input: (W, J, 3) -> Output: (W, J, 9)
        kinematics = compute_kinematics(skel_window)

        # 3. Flatten Skeleton Features
        # (W, J, 9) -> (W, J*9)
        W, J, D = kinematics.shape
        kinematics_flat = kinematics.reshape(W, J * D)

        # 4. Early Fusion: Concatenate Skeleton + Audio
        # (W, J*9) + (W, N_MFCC) -> (W, Input_Dim)
        features = np.concatenate([kinematics_flat, audio_window], axis=-1)

        # Convert to Tensors
        features_tensor = torch.tensor(features, dtype=torch.float32)
        labels_tensor = torch.tensor(labels_window, dtype=torch.long)

        return features_tensor, labels_tensor
