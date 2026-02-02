import os
import numpy as np
import torch
from torch.utils.data import Dataset
from library.config import (
    WINDOW_SIZE,
    STRIDE,
    SKELETON_JOINTS,
    AUDIO_MFCC_DIM,
    INPUT_DIM,
    CACHE_DIR,
)
from library.data_parser import DataParser


class GestureDataset(Dataset):
    """
    PyTorch Dataset for Gesture Recognition.
    Implements Kinematically Consistent Augmentation and Sliding Window logic.
    """

    def __init__(
        self, metadata_path, split_name, load_cache=True, augment=False, debug_size=None
    ):
        """
        Args:
            metadata_path (str): Path to the metadata CSV.
            split_name (str): 'train', 'val', or 'test'.
            load_cache (bool): Whether to load from .npz cache.
            augment (bool): Whether to apply geometric augmentation.
            debug_size (int, optional): Limit dataset size for debugging.
        """
        self.augment = augment
        self.split_name = split_name

        # 1. Load Data via DataParser
        # This returns a dict with 'skeletons', 'audio', 'labels', 'lengths', 'sample_ids'
        # Skeletons shape: (N, MaxLen, 20, 3)
        # Audio shape: (N, MaxLen, 13)
        data_dict = DataParser.process_dataset(
            metadata_path, split_name, load_cache=load_cache, debug_size=debug_size
        )

        if not data_dict:
            # Handle empty dataset case
            self.skeletons = np.zeros((0, 0, 20, 3), dtype=np.float32)
            self.audio = np.zeros((0, 0, 13), dtype=np.float32)
            self.labels = np.zeros((0, 0), dtype=np.int32)
            self.lengths = np.zeros((0,), dtype=np.int32)
            self.sample_ids = np.array([])
            self.windows = []
            return

        self.skeletons = data_dict["skeletons"]
        self.audio = data_dict["audio"]
        self.labels = data_dict["labels"]
        self.lengths = data_dict["lengths"]
        self.sample_ids = data_dict["sample_ids"]

        # 2. Pre-calculate Sliding Windows
        # List of tuples: (sample_idx, start_frame, end_frame)
        self.windows = []

        num_samples = len(self.sample_ids)
        for idx in range(num_samples):
            valid_len = self.lengths[idx]

            # If sequence is shorter than window, take one window with padding
            if valid_len <= WINDOW_SIZE:
                self.windows.append((idx, 0, valid_len))
            else:
                # Generate sliding windows
                # We want to cover the whole sequence.
                # Standard sliding:
                for start in range(0, valid_len - WINDOW_SIZE + 1, STRIDE):
                    end = start + WINDOW_SIZE
                    self.windows.append((idx, start, end))

                # Handle the remainder if the last window didn't reach the end
                # and we want full coverage (especially for test)
                last_start = range(0, valid_len - WINDOW_SIZE + 1, STRIDE)[-1]
                if last_start + WINDOW_SIZE < valid_len:
                    # Add a final window aligned to the end of the sequence
                    # This ensures the last frames are seen
                    start = valid_len - WINDOW_SIZE
                    end = valid_len
                    self.windows.append((idx, start, end))

    def __len__(self):
        return len(self.windows)

    def _apply_augmentation(self, positions):
        """
        Applies random rotation (Y-axis) and scaling to 3D positions.
        positions: (T, 20, 3)
        """
        # Random Rotation around Y-axis
        theta = np.random.uniform(-np.pi / 6, np.pi / 6)  # +/- 30 degrees
        c, s = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

        # Random Scaling (0.8x to 1.2x)
        scale = np.random.uniform(0.8, 1.2)

        # Apply rotation: (T, J, 3) @ (3, 3) -> (T, J, 3)
        # Reshape to (T*J, 3) for matmul
        T, J, C = positions.shape
        flat_pos = positions.reshape(-1, 3)
        rotated_pos = flat_pos @ rotation_matrix.T

        # Apply scaling
        augmented_pos = rotated_pos * scale

        return augmented_pos.reshape(T, J, C)

    def _compute_kinematics(self, positions):
        """
        Computes Velocity and Acceleration from positions.
        positions: (T, 20, 3)
        Returns: (T, 20, 9) -> [Pos, Vel, Acc]
        """
        # Velocity: P[t] - P[t-1]
        # Pad first frame with 0
        vel = np.zeros_like(positions)
        vel[1:] = positions[1:] - positions[:-1]

        # Acceleration: V[t] - V[t-1]
        acc = np.zeros_like(positions)
        acc[1:] = vel[1:] - vel[:-1]

        # Concatenate: (T, 20, 3) x 3 -> (T, 20, 9)
        kinematics = np.concatenate([positions, vel, acc], axis=-1)
        return kinematics

    def __getitem__(self, idx):
        sample_idx, start, end = self.windows[idx]

        # 1. Extract Raw Data
        # Skeletons: (MaxLen, 20, 3) -> slice -> (SeqLen, 20, 3)
        raw_pos = self.skeletons[sample_idx, start:end].copy()
        raw_audio = self.audio[sample_idx, start:end].copy()
        raw_labels = self.labels[sample_idx, start:end].copy()

        seq_len = raw_pos.shape[0]

        # 2. Augmentation (Training Only)
        # Apply geometric transforms to raw positions BEFORE computing derivatives
        if self.augment:
            pos_processed = self._apply_augmentation(raw_pos)
        else:
            pos_processed = raw_pos

        # 3. Compute Kinematics (Physical Validity)
        # Returns (SeqLen, 20, 9)
        # Normalize coordinates slightly to help convergence (mm -> m approx)
        pos_processed = pos_processed / 1000.0
        skel_features = self._compute_kinematics(pos_processed)

        # Flatten joints: (SeqLen, 20, 9) -> (SeqLen, 180)
        skel_features_flat = skel_features.reshape(seq_len, -1)

        # 4. Feature Fusion
        # Concatenate Skeleton (180) + Audio (13) -> (SeqLen, 193)
        fused_features = np.concatenate([skel_features_flat, raw_audio], axis=-1)

        # 5. Padding (if sequence < WINDOW_SIZE)
        # We need fixed size output
        if seq_len < WINDOW_SIZE:
            pad_len = WINDOW_SIZE - seq_len
            # Pad features with 0
            feat_pad = np.zeros((pad_len, fused_features.shape[1]), dtype=np.float32)
            fused_features = np.concatenate([fused_features, feat_pad], axis=0)

            # Pad labels with 0 (Background)
            label_pad = np.zeros((pad_len,), dtype=np.int32)
            raw_labels = np.concatenate([raw_labels, label_pad], axis=0)

        # 6. Convert to Tensor
        # Input: (WindowSize, InputDim)
        # Target: (WindowSize)
        x = torch.from_numpy(fused_features).float()
        y = torch.from_numpy(raw_labels).long()

        # Return sample_idx for tracking during inference
        return x, y, sample_idx
