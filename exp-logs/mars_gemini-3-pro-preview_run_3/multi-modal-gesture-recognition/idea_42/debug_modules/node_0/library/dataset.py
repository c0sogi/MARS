import os
import json
import numpy as np
import pandas as pd
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
from library.config import Config
from library.utils import load_mat_robust


class GestureDataset(Dataset):
    def __init__(self, split="train", load_cached_data=True):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use cached .npz files.
        """
        self.split = split
        self.cache_dir = Config.WORK_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

        # Seed for reproducibility
        self.rng = np.random.default_rng(Config.SEED)

        # Metadata path
        if split == "train":
            meta_path = os.path.join(Config.METADATA_DIR, "train.csv")
        elif split == "val":
            meta_path = os.path.join(Config.METADATA_DIR, "val.csv")
        else:
            meta_path = os.path.join(Config.METADATA_DIR, "test.csv")

        self.meta_df = pd.read_csv(meta_path)

        # Cache file path
        self.cache_file = os.path.join(self.cache_dir, f"dataset_{split}.npz")

        # Load or Create Data
        if load_cached_data and os.path.exists(self.cache_file):
            self._load_cache()
        else:
            self._create_cache()

        # Build Index Map (Windowing vs Full Sequence)
        self.indices = []
        self._build_indices()

    def _create_cache(self):
        """
        Loads raw files, processes them into flat arrays, and saves to .npz.
        """
        print(f"[{self.split}] Creating cache at {self.cache_file}...")

        all_skeleton = []
        all_audio = []
        all_labels = []
        boundaries = []
        sample_ids = []

        current_idx = 0

        for _, row in self.meta_df.iterrows():
            sample_id = row["sample_id"]
            data_path = os.path.join(Config.INPUT_DIR, row["data_path"])
            audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])

            # 1. Load Skeleton
            # Shape: (T, Joints, 3)
            skel_data = load_mat_robust(data_path)

            if skel_data is None:
                # Fallback for corrupt/missing skeleton: create dummy
                # Try to infer length from audio or default to small size
                # This is a rare edge case.
                print(f"Warning: Failed to load skeleton for {sample_id}. Using dummy.")
                T_dummy = 100
                skel_data = np.zeros((T_dummy, Config.NUM_JOINTS, 3), dtype=np.float32)

            num_frames = skel_data.shape[0]

            # 2. Load Audio & Align
            # Shape: (T, 13)
            audio_features = self._process_audio(audio_path, num_frames)

            # 3. Process Labels
            # Shape: (T,)
            label_seq = np.zeros(num_frames, dtype=np.int32)  # Default 0 (Background)
            if isinstance(row["labels"], str):
                labels_list = json.loads(row["labels"])
                for l in labels_list:
                    # Metadata uses 1-based indexing for frames usually, or 0?
                    # Assuming standard python 0-based or converting.
                    # MATLAB usually 1-based. Let's assume inclusive [begin, end].
                    # We clamp to valid range.
                    start = max(0, int(l["begin"]) - 1)  # Adjust if 1-based
                    end = min(num_frames, int(l["end"]))
                    lid = int(l["id"])
                    if start < end:
                        label_seq[start:end] = lid

            # Append to lists
            all_skeleton.append(skel_data)
            all_audio.append(audio_features)
            all_labels.append(label_seq)
            sample_ids.append(sample_id)

            # Record boundary
            boundaries.append([current_idx, current_idx + num_frames])
            current_idx += num_frames

        # Concatenate into flat arrays
        # Use float32 for features, int32 for labels
        flat_skeleton = np.concatenate(all_skeleton, axis=0).astype(np.float32)
        flat_audio = np.concatenate(all_audio, axis=0).astype(np.float32)
        flat_labels = np.concatenate(all_labels, axis=0).astype(np.int32)
        arr_boundaries = np.array(boundaries, dtype=np.int32)
        arr_sample_ids = np.array(sample_ids)  # numpy handles strings

        np.savez_compressed(
            self.cache_file,
            skeleton=flat_skeleton,
            audio=flat_audio,
            labels=flat_labels,
            boundaries=arr_boundaries,
            sample_ids=arr_sample_ids,
        )

        # Load into memory
        self.flat_skeleton = flat_skeleton
        self.flat_audio = flat_audio
        self.flat_labels = flat_labels
        self.boundaries = arr_boundaries
        self.sample_ids = arr_sample_ids

    def _load_cache(self):
        """Loads data from .npz file."""
        print(f"[{self.split}] Loading cache from {self.cache_file}...")
        data = np.load(self.cache_file, allow_pickle=True)
        self.flat_skeleton = data["skeleton"]
        self.flat_audio = data["audio"]
        self.flat_labels = data["labels"]
        self.boundaries = data["boundaries"]
        self.sample_ids = data["sample_ids"]

    def _process_audio(self, audio_path, target_frames):
        """
        Loads audio, computes MFCC, and aligns to target_frames.
        Returns: (target_frames, n_mfcc)
        """
        if not os.path.exists(audio_path):
            return np.zeros((target_frames, Config.AUDIO_N_MFCC), dtype=np.float32)

        try:
            waveform, sample_rate = torchaudio.load(audio_path)

            # Compute MFCC
            mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=sample_rate,
                n_mfcc=Config.AUDIO_N_MFCC,
                melkwargs={
                    "n_fft": 400,
                    "hop_length": 160,
                    "n_mels": 23,
                    "center": False,
                },
            )

            # (Channel, n_mfcc, time)
            mfcc = mfcc_transform(waveform)

            # Average over channels if stereo
            if mfcc.shape[0] > 1:
                mfcc = torch.mean(mfcc, dim=0, keepdim=True)

            # Interpolate to match video frames
            # Input to interpolate must be (Batch, Channels, Time)
            # Current: (1, n_mfcc, time_audio)
            mfcc = F.interpolate(
                mfcc, size=target_frames, mode="linear", align_corners=False
            )

            # Shape: (1, n_mfcc, target_frames) -> (target_frames, n_mfcc)
            mfcc = mfcc.squeeze(0).permute(1, 0).numpy()

            return mfcc.astype(np.float32)

        except Exception as e:
            # print(f"Audio error: {e}")
            return np.zeros((target_frames, Config.AUDIO_N_MFCC), dtype=np.float32)

    def _build_indices(self):
        """
        Constructs the list of items to retrieve.
        Train: Sliding windows.
        Val/Test: Full sequences.
        """
        self.indices = []

        for i, (start, end) in enumerate(self.boundaries):
            seq_len = end - start

            if self.split == "train":
                # Sliding Window
                # If sequence is shorter than window, pad it (handled in getitem via slicing logic)
                # or just take one window.
                if seq_len <= Config.WINDOW_SIZE:
                    self.indices.append(
                        (i, 0, seq_len)
                    )  # Sample idx, relative start, length
                else:
                    # Stride
                    for t in range(0, seq_len - Config.WINDOW_SIZE + 1, Config.STRIDE):
                        self.indices.append((i, t, Config.WINDOW_SIZE))

                    # Ensure last frame is covered if not exact fit
                    if (seq_len - Config.WINDOW_SIZE) % Config.STRIDE != 0:
                        self.indices.append(
                            (i, seq_len - Config.WINDOW_SIZE, Config.WINDOW_SIZE)
                        )
            else:
                # Full Sequence
                self.indices.append((i, 0, seq_len))

    def __len__(self):
        return len(self.indices)

    def _augment_skeleton(self, pos):
        """
        Applies random rotation (Y-axis) and scaling to positions.
        pos: (T, J, 3)
        """
        # 1. Random Scale
        scale = self.rng.uniform(0.9, 1.1)
        pos = pos * scale

        # 2. Random Rotation around Y-axis
        # Angle in radians. +/- 15 degrees = +/- 0.26 rad
        theta = self.rng.uniform(-0.26, 0.26)
        c, s = np.cos(theta), np.sin(theta)

        # Rotation matrix for Y-axis
        # x' = x*c + z*s
        # y' = y
        # z' = -x*s + z*c

        x = pos[:, :, 0]
        y = pos[:, :, 1]
        z = pos[:, :, 2]

        new_x = x * c + z * s
        new_z = -x * s + z * c

        pos_aug = np.stack([new_x, y, new_z], axis=2)
        return pos_aug

    def _compute_kinematics(self, pos):
        """
        Computes Velocity and Acceleration from Positions.
        pos: (T, J, 3)
        Returns: (T, J*9) flattened vector [Pos, Vel, Acc]
        """
        T, J, _ = pos.shape

        # Velocity: P_t - P_{t-1}
        vel = np.zeros_like(pos)
        vel[1:] = pos[1:] - pos[:-1]
        vel[0] = vel[1]  # Pad start

        # Acceleration: V_t - V_{t-1}
        acc = np.zeros_like(vel)
        acc[1:] = vel[1:] - vel[:-1]
        acc[0] = acc[1]  # Pad start

        # Concatenate: (T, J, 9)
        features = np.concatenate([pos, vel, acc], axis=2)

        # Flatten joints: (T, J*9)
        features_flat = features.reshape(T, -1)

        return features_flat

    def __getitem__(self, idx):
        sample_idx, rel_start, length = self.indices[idx]

        # Global boundaries
        global_start_idx = self.boundaries[sample_idx, 0]

        # Absolute slice indices
        abs_start = global_start_idx + rel_start
        abs_end = abs_start + length

        # 1. Retrieve Raw Data
        # Copy to ensure we don't modify cache when augmenting
        raw_pos = self.flat_skeleton[abs_start:abs_end].copy()  # (T, J, 3)
        audio = self.flat_audio[abs_start:abs_end].copy()  # (T, 13)
        labels = self.flat_labels[abs_start:abs_end].copy()  # (T,)

        # 2. Augmentation (Train only)
        if self.split == "train":
            pos = self._augment_skeleton(raw_pos)
        else:
            pos = raw_pos

        # 3. Kinematics
        # (T, J*9)
        skel_features = self._compute_kinematics(pos)

        # 4. Fusion
        # (T, J*9 + 13)
        combined_features = np.concatenate([skel_features, audio], axis=1)

        # 5. Convert to Tensor
        x = torch.from_numpy(combined_features).float()
        y = torch.from_numpy(labels).long()

        # Metadata
        sample_id = str(self.sample_ids[sample_idx])

        return x, y, sample_id
