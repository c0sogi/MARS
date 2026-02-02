import os
import json
import numpy as np
import pandas as pd
import torch
import librosa
from torch.utils.data import Dataset
from library.config import Config
from library.utils import robust_load_mat


class GestureDataset(Dataset):
    def __init__(self, split, load_cached_data=True, mode="train"):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use cached pre-processed raw data.
            mode (str): 'train' (returns fixed windows) or 'inference' (returns full sequences).
        """
        self.split = split
        self.mode = mode
        self.rng = np.random.default_rng(Config.SEED)

        # Select metadata file
        if split == "train":
            self.metadata_file = Config.TRAIN_CSV
        elif split == "val":
            self.metadata_file = Config.VAL_CSV
        elif split == "test":
            self.metadata_file = Config.TEST_CSV
        else:
            raise ValueError(f"Unknown split: {split}")

        # Define cache path
        self.cache_path = os.path.join(Config.CACHE_DIR, f"dataset_{split}.npz")

        # Load Raw Aligned Data (Skeleton, Audio, Labels)
        self.data = self._load_data(load_cached_data)

        # Pre-calculate windows for training
        if self.mode == "train":
            self.windows = self._prepare_windows()

        # Define Bone Connectivity (Child -> Parent) for Feature Derivation
        # 0:HipCenter is used as Root.
        self.bone_pairs = [
            (1, 0),
            (2, 1),
            (3, 2),
            (4, 2),
            (5, 4),
            (6, 5),
            (7, 6),
            (8, 2),
            (9, 8),
            (10, 9),
            (11, 10),
            (12, 0),
            (13, 12),
            (14, 13),
            (15, 14),
            (16, 0),
            (17, 16),
            (18, 17),
            (19, 18),
        ]

    def _load_data(self, load_cached_data):
        """Loads data from cache or computes it from scratch."""
        # 1. Try Loading Cache
        if load_cached_data and os.path.exists(self.cache_path):
            try:
                print(f"Loading cached data from {self.cache_path}...")
                loaded = np.load(self.cache_path, allow_pickle=True)
                ids = loaded["ids"]
                data_list = []
                # Reconstruct list of dicts from flattened arrays
                for i in range(len(ids)):
                    data_list.append(
                        {
                            "sample_id": str(ids[i]),
                            "skeleton": loaded[f"skeleton_{i}"],
                            "audio": loaded[f"audio_{i}"],
                            "labels": loaded[f"labels_{i}"],
                        }
                    )
                return data_list
            except Exception as e:
                print(f"Cache load failed: {e}. Recomputing...")

        # 2. Compute from Scratch
        print(f"Processing data for split: {self.split}...")
        df = pd.read_csv(self.metadata_file)
        data_list = []

        for _, row in df.iterrows():
            sample_id = row["sample_id"]
            skel_path = os.path.join(Config.INPUT_DIR, row["data_path"])
            audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])

            # Load Skeleton
            skeleton = robust_load_mat(skel_path)  # (T, 20, 3)
            num_frames = skeleton.shape[0]

            # Load & Align Audio
            audio_features = self._process_audio(audio_path, num_frames)

            # Process Labels (Dense Frame-wise)
            labels = np.zeros(num_frames, dtype=np.int64)
            if self.split != "test":
                try:
                    label_list = json.loads(row["labels"])
                    for l in label_list:
                        gid = int(l["id"])
                        # Convert 1-based indexing to 0-based
                        start = max(0, int(l["begin"]) - 1)
                        end = min(num_frames, int(l["end"]))
                        if start < end:
                            labels[start:end] = gid
                except Exception:
                    pass  # Keep as background if parsing fails

            data_list.append(
                {
                    "sample_id": sample_id,
                    "skeleton": skeleton.astype(np.float32),
                    "audio": audio_features.astype(np.float32),
                    "labels": labels,
                }
            )

        # 3. Save to Cache
        # We store variable length arrays as separate entries in the npz
        save_dict = {"ids": np.array([d["sample_id"] for d in data_list])}
        for i, d in enumerate(data_list):
            save_dict[f"skeleton_{i}"] = d["skeleton"]
            save_dict[f"audio_{i}"] = d["audio"]
            save_dict[f"labels_{i}"] = d["labels"]

        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        np.savez_compressed(self.cache_path, **save_dict)
        print(f"Cached data saved to {self.cache_path}")

        return data_list

    def _process_audio(self, audio_path, target_frames):
        """Computes MFCCs and aligns them to the video frame count."""
        if not os.path.exists(audio_path) or target_frames == 0:
            return np.zeros((target_frames, Config.AUDIO_MFCC_N_MFCC), dtype=np.float32)

        try:
            y, sr = librosa.load(audio_path, sr=Config.AUDIO_SAMPLE_RATE)
            # Compute MFCC
            mfcc = librosa.feature.mfcc(
                y=y, sr=sr, n_mfcc=Config.AUDIO_MFCC_N_MFCC, n_fft=2048, hop_length=512
            ).T  # (Time, n_mfcc)

            # Resample to match video frames
            curr_len = mfcc.shape[0]
            if curr_len != target_frames:
                resampled = np.zeros(
                    (target_frames, Config.AUDIO_MFCC_N_MFCC), dtype=np.float32
                )
                x_old = np.linspace(0, 1, curr_len)
                x_new = np.linspace(0, 1, target_frames)
                for c in range(Config.AUDIO_MFCC_N_MFCC):
                    resampled[:, c] = np.interp(x_new, x_old, mfcc[:, c])
                return resampled
            return mfcc
        except Exception:
            return np.zeros((target_frames, Config.AUDIO_MFCC_N_MFCC), dtype=np.float32)

    def _prepare_windows(self):
        """Generates sliding window indices for training."""
        windows = []
        for i, sample in enumerate(self.data):
            num_frames = sample["skeleton"].shape[0]
            if num_frames < Config.WINDOW_SIZE:
                windows.append((i, 0))
            else:
                # Sliding window with overlap
                for start in range(
                    0, num_frames - Config.WINDOW_SIZE + 1, Config.STRIDE
                ):
                    windows.append((i, start))
                # Ensure coverage of the tail
                last_start = num_frames - Config.WINDOW_SIZE
                if last_start > 0 and (last_start % Config.STRIDE != 0):
                    windows.append((i, last_start))
        return windows

    def _augment_skeleton(self, skeleton):
        """Applies random Y-axis rotation and scaling."""
        # Scaling
        scale = self.rng.uniform(0.9, 1.1)
        skeleton = skeleton * scale

        # Rotation
        angle_deg = self.rng.uniform(-15, 15)
        angle_rad = np.radians(angle_deg)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        R_y = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

        # Apply rotation
        shape = skeleton.shape
        flat = skeleton.reshape(-1, 3)
        rotated = flat @ R_y.T
        return rotated.reshape(shape)

    def _compute_features(self, skeleton, audio):
        """Derives explicit spatial-temporal features."""
        T, J, C = skeleton.shape

        # 1. Root-Relative Positions (Root = HipCenter/Index 0)
        root = skeleton[:, 0:1, :]
        rel_pos = skeleton - root

        # 2. Bone Vectors
        bone_vecs = np.zeros_like(skeleton)
        for child, parent in self.bone_pairs:
            bone_vecs[:, child, :] = skeleton[:, child, :] - skeleton[:, parent, :]

        # 3. Velocity
        velocity = np.zeros_like(rel_pos)
        velocity[1:] = rel_pos[1:] - rel_pos[:-1]

        # 4. Acceleration
        accel = np.zeros_like(velocity)
        accel[1:] = velocity[1:] - velocity[:-1]

        # Flatten and Concatenate
        features = np.concatenate(
            [
                rel_pos.reshape(T, -1),
                bone_vecs.reshape(T, -1),
                velocity.reshape(T, -1),
                accel.reshape(T, -1),
                audio,
            ],
            axis=1,
        )

        return features

    def __len__(self):
        if self.mode == "train":
            return len(self.windows)
        return len(self.data)

    def __getitem__(self, idx):
        if self.mode == "train":
            # Training Mode: Return fixed-size window
            sample_idx, start_frame = self.windows[idx]
            sample = self.data[sample_idx]

            raw_skel = sample["skeleton"]
            raw_audio = sample["audio"]
            raw_labels = sample["labels"]

            total_frames = raw_skel.shape[0]

            # Handle short sequences (Padding)
            if total_frames < Config.WINDOW_SIZE:
                pad_len = Config.WINDOW_SIZE - total_frames

                # Augment full sequence
                aug_skel = self._augment_skeleton(raw_skel)
                feats = self._compute_features(aug_skel, raw_audio)

                # Pad
                feat_pad = np.zeros((pad_len, feats.shape[1]), dtype=np.float32)
                features = np.concatenate([feats, feat_pad], axis=0)

                label_pad = np.zeros(pad_len, dtype=np.int64)
                labels = np.concatenate([raw_labels, label_pad], axis=0)
            else:
                # Slice window
                end_frame = start_frame + Config.WINDOW_SIZE
                skel_slice = raw_skel[start_frame:end_frame]
                audio_slice = raw_audio[start_frame:end_frame]
                labels = raw_labels[start_frame:end_frame]

                # Augment slice
                aug_skel = self._augment_skeleton(skel_slice)
                features = self._compute_features(aug_skel, audio_slice)

            return torch.from_numpy(features).float(), torch.from_numpy(labels).long()

        else:
            # Inference Mode: Return full sequence
            sample = self.data[idx]
            # No augmentation
            features = self._compute_features(sample["skeleton"], sample["audio"])
            labels = sample["labels"]
            return (
                torch.from_numpy(features).float(),
                torch.from_numpy(labels).long(),
                sample["sample_id"],
            )
