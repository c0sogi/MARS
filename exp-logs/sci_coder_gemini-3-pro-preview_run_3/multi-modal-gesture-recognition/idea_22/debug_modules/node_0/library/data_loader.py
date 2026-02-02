import os
import json
import numpy as np
import pandas as pd
import torch
import librosa
from scipy.interpolate import interp1d
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import load_mat_safe


class GestureDataset(Dataset):
    """
    Dataset class for the Residual Log-Kinematic Refinement Network.
    Handles loading, caching, kinematically consistent augmentation, and windowing.
    """

    def __init__(self, mode, load_cached_data=True, transform=False):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use cached .npz files.
            transform (bool): Whether to apply augmentation (Rotation/Scaling).
        """
        self.mode = mode
        self.transform = transform
        self.window_size = Config.WINDOW_SIZE
        self.stride = Config.STRIDE
        self.num_joints = Config.NUM_JOINTS

        # Load data (Cached or Processed)
        self.skel_data, self.audio_data, self.labels, self.boundaries = self._load_data(
            load_cached_data
        )

        # Generate window indices
        self.window_indices = self._make_window_indices()

    def _load_data(self, load_cached):
        """
        Loads data from cache or processes it from scratch.
        Returns concatenated arrays and boundaries.
        """
        cache_path = os.path.join(Config.CACHE_DIR, f"dataset_{self.mode}.npz")

        if load_cached and os.path.exists(cache_path):
            print(f"Loading cached {self.mode} data from {cache_path}...")
            try:
                data = np.load(cache_path)
                return data["skel"], data["audio"], data["labels"], data["boundaries"]
            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing...")

        # Process from scratch
        print(f"Processing {self.mode} data...")

        # Load Metadata
        if self.mode == "train":
            meta_path = Config.TRAIN_METADATA_PATH
        elif self.mode == "val":
            meta_path = Config.VAL_METADATA_PATH
        else:
            meta_path = Config.TEST_METADATA_PATH

        df = pd.read_csv(meta_path)

        # Debug subset
        if Config.DEBUG:
            df = df.iloc[: Config.DEBUG_SUBSET_SIZE]

        all_skel = []
        all_audio = []
        all_labels = []
        boundaries = [0]

        for _, row in df.iterrows():
            # 1. Load Skeleton
            mat_path = os.path.join(Config.INPUT_DIR, row["data_path"])
            skel = load_mat_safe(mat_path)  # (T, 20, 3)

            if skel is None:
                continue

            num_frames = skel.shape[0]
            if num_frames < self.window_size:
                continue

            # Flatten skeleton: (T, 20, 3) -> (T, 60)
            # We keep it as (T, 20, 3) for now to allow 3D rotation in __getitem__
            # But for caching efficiency and simplicity in _process, let's keep (T, 20, 3)

            # 2. Load Audio & Align
            audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])
            audio_features = self._process_audio(audio_path, num_frames)

            # 3. Create Labels
            label_seq = np.zeros(num_frames, dtype=np.int64)
            if self.mode != "test":
                label_list = json.loads(row["labels"])
                for l in label_list:
                    # Ensure indices are within bounds
                    start = max(0, l["begin"] - 1)  # 1-based to 0-based
                    end = min(num_frames, l["end"])
                    label_seq[start:end] = l["id"]

            all_skel.append(skel)
            all_audio.append(audio_features)
            all_labels.append(label_seq)
            boundaries.append(boundaries[-1] + num_frames)

        if len(all_skel) == 0:
            raise RuntimeError("No valid data found after processing.")

        # Concatenate
        # Skel: (TotalFrames, 20, 3)
        # Audio: (TotalFrames, 13)
        # Labels: (TotalFrames,)
        cat_skel = np.concatenate(all_skel, axis=0).astype(np.float32)
        cat_audio = np.concatenate(all_audio, axis=0).astype(np.float32)
        cat_labels = np.concatenate(all_labels, axis=0).astype(np.int64)
        boundaries = np.array(boundaries, dtype=np.int64)

        # Save to cache
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        np.savez(
            cache_path,
            skel=cat_skel,
            audio=cat_audio,
            labels=cat_labels,
            boundaries=boundaries,
        )
        print(f"Saved {self.mode} data to {cache_path}")

        return cat_skel, cat_audio, cat_labels, boundaries

    def _process_audio(self, audio_path, target_frames):
        """
        Loads audio, extracts MFCCs, and resamples to match target_frames.
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=Config.AUDIO_SAMPLE_RATE)
            if len(y) == 0:
                return np.zeros((target_frames, Config.AUDIO_N_MFCC), dtype=np.float32)

            # Extract MFCC
            # Hop length default is 512.
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=Config.AUDIO_N_MFCC)
            # mfcc shape: (n_mfcc, n_audio_frames)

            n_mfcc, n_audio_frames = mfcc.shape

            # Interpolate to match video frames
            x_old = np.linspace(0, 1, n_audio_frames)
            x_new = np.linspace(0, 1, target_frames)

            f = interp1d(x_old, mfcc, axis=1, kind="linear", fill_value="extrapolate")
            mfcc_resampled = f(x_new).T  # (target_frames, n_mfcc)

            return mfcc_resampled.astype(np.float32)

        except Exception:
            # Fallback for missing/corrupt audio
            return np.zeros((target_frames, Config.AUDIO_N_MFCC), dtype=np.float32)

    def _make_window_indices(self):
        """
        Creates a list of (start_idx, end_idx, sample_idx) for all valid sliding windows.
        """
        indices = []
        num_samples = len(self.boundaries) - 1

        for i in range(num_samples):
            start_boundary = self.boundaries[i]
            end_boundary = self.boundaries[i + 1]
            length = end_boundary - start_boundary

            if length < self.window_size:
                continue

            # Generate start offsets relative to the sample
            # range(0, length - window_size + 1, stride)
            for rel_start in range(0, length - self.window_size + 1, self.stride):
                global_start = start_boundary + rel_start
                global_end = global_start + self.window_size
                indices.append((global_start, global_end, i))

        return indices

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        global_start, global_end, sample_idx = self.window_indices[idx]
        sample_boundary_start = self.boundaries[sample_idx]

        # -----------------------------------------------------------
        # 1. Fetch Raw Data with Context for Derivatives
        # -----------------------------------------------------------
        # We need 2 extra frames of context at the start to compute
        # acceleration for the first frame of the window.
        # Window: [t, t+1, ..., t+63]
        # Need: [t-2, t-1, t, ..., t+63]

        req_start = global_start - 2
        req_end = global_end

        # Handle boundary padding
        pad_front = 0
        if req_start < sample_boundary_start:
            pad_front = sample_boundary_start - req_start
            req_start = sample_boundary_start

        # Slice raw skeleton: (T_slice, 20, 3)
        raw_skel = self.skel_data[req_start:req_end].copy()

        # Pad if necessary (repeat first frame)
        if pad_front > 0:
            first_frame = raw_skel[0:1]  # (1, 20, 3)
            padding = np.repeat(first_frame, pad_front, axis=0)
            raw_skel = np.concatenate([padding, raw_skel], axis=0)

        # -----------------------------------------------------------
        # 2. Kinematically Consistent Augmentation
        # -----------------------------------------------------------
        # Apply augmentation to Position BEFORE computing derivatives

        # Normalize units: mm -> meters
        raw_skel = raw_skel / 1000.0

        if self.transform:
            # Random Rotation around Y-axis
            theta = np.deg2rad(np.random.uniform(-15, 15))
            c, s = np.cos(theta), np.sin(theta)
            # Rotation matrix for Y-axis
            R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

            # Apply rotation: P_aug = P @ R.T
            # raw_skel shape: (T, Joints, 3)
            # Reshape to (T*Joints, 3) for matmul
            T, J, C = raw_skel.shape
            flat_skel = raw_skel.reshape(-1, 3)
            flat_skel = flat_skel @ R.T
            raw_skel = flat_skel.reshape(T, J, C)

            # Random Scaling
            scale = np.random.uniform(0.9, 1.1)
            raw_skel = raw_skel * scale

        # -----------------------------------------------------------
        # 3. Compute Derivatives (Velocity & Acceleration)
        # -----------------------------------------------------------
        # raw_skel has shape (Window+2, 20, 3)

        # Velocity: V[t] = P[t] - P[t-1]
        # We use numpy diff.
        # diff(P, axis=0) gives size (Window+1, 20, 3)
        velocity = np.diff(raw_skel, axis=0)

        # Acceleration: A[t] = V[t] - V[t-1]
        # diff(V, axis=0) gives size (Window, 20, 3)
        acceleration = np.diff(velocity, axis=0)

        # We now have:
        # P: raw_skel (Window+2) -> need last Window
        # V: velocity (Window+1) -> need last Window
        # A: acceleration (Window) -> is exactly Window

        pos_feat = raw_skel[2:]  # (Window, 20, 3)
        vel_feat = velocity[1:]  # (Window, 20, 3)
        acc_feat = acceleration  # (Window, 20, 3)

        # Flatten joints: (Window, 60)
        pos_feat = pos_feat.reshape(self.window_size, -1)
        vel_feat = vel_feat.reshape(self.window_size, -1)
        acc_feat = acc_feat.reshape(self.window_size, -1)

        # -----------------------------------------------------------
        # 4. Audio & Fusion
        # -----------------------------------------------------------
        # Fetch audio for the exact window (no context needed for MFCC as it's pre-computed)
        audio_feat = self.audio_data[global_start:global_end]  # (Window, 13)

        # Concatenate all features
        # [Pos(60), Vel(60), Acc(60), Audio(13)] -> (Window, 193)
        features = np.concatenate([pos_feat, vel_feat, acc_feat, audio_feat], axis=1)

        # -----------------------------------------------------------
        # 5. Labels
        # -----------------------------------------------------------
        labels = self.labels[global_start:global_end]  # (Window,)

        return torch.from_numpy(features).float(), torch.from_numpy(labels).long()


def get_dataloaders(batch_size=Config.BATCH_SIZE, num_workers=2):
    """
    Factory function to create DataLoaders for train, val, and test.
    """
    # Create Datasets
    # Train: Augmentation = True
    train_ds = GestureDataset(mode="train", load_cached_data=True, transform=True)

    # Val/Test: Augmentation = False
    val_ds = GestureDataset(mode="val", load_cached_data=True, transform=False)
    test_ds = GestureDataset(mode="test", load_cached_data=True, transform=False)

    # Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batch for training stability
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
