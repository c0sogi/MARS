import os
import sys
import json
import numpy as np
import pandas as pd
import scipy.io
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation as R

# Add library path to access config
sys.path.append("./library")
from config import Config


class ItalianGestureDataset(Dataset):
    def __init__(
        self, metadata_path, split="train", load_cached_data=True, augment=False
    ):
        """
        Args:
            metadata_path (str): Path to the CSV metadata file.
            split (str): 'train', 'val', or 'test'. Used for cache naming.
            load_cached_data (bool): Whether to load from cache if available.
            augment (bool): Whether to apply kinematic augmentation.
        """
        self.split = split
        self.augment = augment
        self.window_size = Config.WINDOW_SIZE
        self.stride = Config.STRIDE

        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        # Cache file path
        self.cache_path = os.path.join(Config.CACHE_DIR, f"dataset_{split}.npz")

        # Load Data (Cached or Raw)
        if load_cached_data and os.path.exists(self.cache_path):
            print(f"Loading cached {split} data from {self.cache_path}...")
            self.data = self._load_cache()
        else:
            print(f"Processing {split} data from scratch...")
            self.data = self._process_raw_data(metadata_path)
            self._save_cache(self.data)

        # Generate Windows
        self.windows = self._create_windows()

        print(
            f"Dataset {split} initialized. Num samples: {len(self.data)}, Num windows: {len(self.windows)}"
        )

    def _process_raw_data(self, metadata_path):
        """Reads raw files and processes them into a dictionary."""
        df = pd.read_csv(metadata_path)
        processed_data = {}

        for _, row in df.iterrows():
            sample_id = row["sample_id"]
            mat_path = os.path.join(Config.INPUT_DIR, row["data_path"])
            audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])

            # Parse labels
            labels = json.loads(row["labels"]) if isinstance(row["labels"], str) else []

            # 1. Load Skeleton Data
            # Returns (T, NumJoints, 3)
            skeleton_pos = self._load_skeleton(mat_path)
            if skeleton_pos is None:
                continue  # Skip corrupt samples

            num_frames = skeleton_pos.shape[0]

            # 2. Load and Process Audio
            # Returns (T, AudioDim) aligned to num_frames
            audio_features = self._load_audio(audio_path, num_frames)

            # 3. Generate Targets
            # class_labels: (T,), boundary_labels: (T,)
            class_labels, boundary_labels = self._generate_targets(labels, num_frames)

            processed_data[sample_id] = {
                "pos": skeleton_pos.astype(np.float32),
                "audio": audio_features.astype(np.float32),
                "cls": class_labels.astype(np.int64),
                "bnd": boundary_labels.astype(np.float32),
            }

        return processed_data

    def _load_skeleton(self, mat_path):
        try:
            mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
            if "Video" not in mat._fieldnames:
                return None
            video = mat.Video
            if not hasattr(video, "Frames"):
                return None

            frames = video.Frames
            # frames can be a single object or array of objects
            if not isinstance(frames, np.ndarray):
                frames = [frames]

            # Extract WorldPosition for all frames
            num_frames = len(frames)
            num_joints = Config.NUM_JOINTS

            pos_data = np.zeros((num_frames, num_joints, 3), dtype=np.float32)

            for t, frame in enumerate(frames):
                if hasattr(frame, "Skeleton") and hasattr(
                    frame.Skeleton, "WorldPosition"
                ):
                    wp = frame.Skeleton.WorldPosition
                    # Handle different potential struct formats
                    if isinstance(wp, np.ndarray):
                        # Array of structs
                        for j in range(min(len(wp), num_joints)):
                            joint = wp[j]
                            pos_data[t, j, 0] = joint.X
                            pos_data[t, j, 1] = joint.Y
                            pos_data[t, j, 2] = joint.Z
                    elif hasattr(wp, "X") and isinstance(wp.X, np.ndarray):
                        # Struct of arrays
                        pos_data[t, :, 0] = wp.X
                        pos_data[t, :, 1] = wp.Y
                        pos_data[t, :, 2] = wp.Z
                    else:
                        pass

            return pos_data

        except Exception as e:
            return None

    def _load_audio(self, audio_path, target_frames):
        try:
            waveform, sample_rate = torchaudio.load(audio_path)

            # Resample to 16kHz if needed (standard for MFCC)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=16000
                )
                waveform = resampler(waveform)
                sample_rate = 16000

            # Compute MFCC
            # n_mfcc=13, resulting in 13 coeffs. We want 39 (MFCC+Delta+DeltaDelta)
            mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=sample_rate,
                n_mfcc=Config.N_MFCC,
                melkwargs={
                    "n_fft": 400,
                    "hop_length": 160,
                    "n_mels": 23,
                    "center": False,
                },
            )

            mfcc = mfcc_transform(waveform)  # (Channels, n_mfcc, time)

            # Compute Deltas
            delta = torchaudio.functional.compute_deltas(mfcc)
            delta2 = torchaudio.functional.compute_deltas(delta)

            # Concatenate: (Channels, 39, time)
            features = torch.cat([mfcc, delta, delta2], dim=1)

            # Average over channels if stereo
            features = features.mean(dim=0)  # (39, time)

            # Interpolate to match video frames
            # Input to interpolate needs to be (Batch, Channels, Time)
            features = features.unsqueeze(0)  # (1, 39, time)

            features = F.interpolate(
                features, size=target_frames, mode="linear", align_corners=False
            )

            features = features.squeeze(0).permute(1, 0)  # (T, 39)

            return features.numpy()

        except Exception as e:
            # Return zeros if failed
            return np.zeros((target_frames, Config.AUDIO_INPUT_DIM), dtype=np.float32)

    def _generate_targets(self, labels, num_frames):
        cls_target = np.zeros(num_frames, dtype=np.int64)  # 0 is background
        bnd_target = np.zeros(num_frames, dtype=np.float32)

        radius = Config.BOUNDARY_RADIUS

        for label in labels:
            gid = label["id"]
            start = max(0, label["begin"] - 1)  # 1-based to 0-based
            end = min(num_frames - 1, label["end"] - 1)

            if start > end:
                continue

            # Classification Target
            cls_target[start : end + 1] = gid

            # Boundary Target
            # Start boundary
            s_min = max(0, start - radius)
            s_max = min(num_frames, start + radius + 1)
            bnd_target[s_min:s_max] = 1.0

            # End boundary
            e_min = max(0, end - radius)
            e_max = min(num_frames, end + radius + 1)
            bnd_target[e_min:e_max] = 1.0

        return cls_target, bnd_target

    def _save_cache(self, data):
        # np.savez_compressed expects keyword args
        np.savez_compressed(self.cache_path, data=data)

    def _load_cache(self):
        # Load the object, extract the dictionary
        loaded = np.load(self.cache_path, allow_pickle=True)
        return loaded["data"].item()

    def _create_windows(self):
        windows = []
        for sample_id, content in self.data.items():
            num_frames = content["pos"].shape[0]

            # If sequence is shorter than window, take one window with padding
            if num_frames <= self.window_size:
                windows.append((sample_id, 0, num_frames))
                continue

            # Sliding window
            for start in range(0, num_frames, self.stride):
                end = min(start + self.window_size, num_frames)
                windows.append((sample_id, start, end))

                # If we reached the end, stop
                if end == num_frames:
                    break
        return windows

    def _augment_kinematics(self, pos):
        """
        Applies random rotation and scaling to positions.
        pos: (T, J, 3)
        """
        # 1. Random Scaling
        scale = np.random.uniform(0.9, 1.1)
        pos = pos * scale

        # 2. Random Rotation
        # Random rotation around Y axis (vertical) is most physically plausible for Kinect data
        # but full 3D rotation (small angles) adds robustness.
        angles = np.random.uniform(-15, 15, size=3)  # degrees
        r = R.from_euler("xyz", angles, degrees=True)
        rot_matrix = r.as_matrix().astype(np.float32)  # (3, 3)

        # Apply rotation: (T, J, 3) @ (3, 3) -> (T, J, 3)
        # Reshape to (T*J, 3) for matmul
        T, J, C = pos.shape
        pos_flat = pos.reshape(-1, 3)
        pos_rotated = pos_flat @ rot_matrix.T
        pos = pos_rotated.reshape(T, J, 3)

        return pos

    def _compute_kinematics(self, pos):
        """
        Computes velocity and acceleration.
        pos: (T, J, 3)
        Returns: (T, J*9) flattened features
        """
        # Velocity: P(t) - P(t-1)
        # Pad first frame
        vel = np.diff(pos, axis=0, prepend=pos[0:1])

        # Acceleration: V(t) - V(t-1)
        acc = np.diff(vel, axis=0, prepend=vel[0:1])

        # Normalize/Scale
        # Positions are in mm. Convert to meters for numerical stability.
        pos = pos / 1000.0
        vel = vel / 1000.0
        acc = acc / 1000.0

        # Concatenate: (T, J, 9)
        feats = np.concatenate([pos, vel, acc], axis=2)

        # Flatten joints: (T, J*9)
        T, J, _ = feats.shape
        feats = feats.reshape(T, -1)

        return feats

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        sample_id, start, end = self.windows[idx]
        sample_data = self.data[sample_id]

        # Extract slices
        pos = sample_data["pos"][start:end]  # (T_slice, J, 3)
        audio = sample_data["audio"][start:end]  # (T_slice, 39)
        cls_lbl = sample_data["cls"][start:end]  # (T_slice,)
        bnd_lbl = sample_data["bnd"][start:end]  # (T_slice,)

        # Augmentation (Train only)
        if self.augment:
            pos = self._augment_kinematics(pos)

        # Compute Kinematics (Pos -> Vel -> Acc)
        # Returns (T_slice, 180)
        skel_features = self._compute_kinematics(pos)

        # Concatenate Audio
        # (T_slice, 219)
        features = np.concatenate([skel_features, audio], axis=1)

        # Padding if window is smaller than target size
        T_slice = features.shape[0]
        if T_slice < self.window_size:
            pad_len = self.window_size - T_slice
            # Pad features with 0
            features = np.pad(features, ((0, pad_len), (0, 0)), mode="constant")
            # Pad labels with 0 (background)
            cls_lbl = np.pad(cls_lbl, (0, pad_len), mode="constant", constant_values=0)
            # Pad boundaries with 0
            bnd_lbl = np.pad(bnd_lbl, (0, pad_len), mode="constant", constant_values=0)

        # Convert to tensors
        features = torch.from_numpy(features).float()
        cls_lbl = torch.from_numpy(cls_lbl).long()
        bnd_lbl = torch.from_numpy(bnd_lbl).float()

        return features, cls_lbl, bnd_lbl, sample_id, start


def get_dataloaders():
    """
    Factory function to create dataloaders for train, val, and test.
    """
    # Train Set
    train_dataset = ItalianGestureDataset(
        Config.TRAIN_METADATA_PATH, split="train", load_cached_data=True, augment=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Val Set
    val_dataset = ItalianGestureDataset(
        Config.VAL_METADATA_PATH, split="val", load_cached_data=True, augment=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Test Set
    test_dataset = ItalianGestureDataset(
        Config.TEST_METADATA_PATH, split="test", load_cached_data=True, augment=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
