import os
import json
import numpy as np
import pandas as pd
import scipy.io
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    WINDOW_SIZE,
    STRIDE,
    BATCH_SIZE,
    AUDIO_DIM,
    SKELETON_DIM,
    VELOCITY_DIM,
    set_seed,
)

# Ensure reproducibility
set_seed()


class GestureDataset(Dataset):
    def __init__(
        self,
        metadata_path,
        mode="train",
        load_cached_data=True,
        cache_dir=os.path.join(WORKING_DIR, "cache"),
    ):
        self.mode = mode
        self.metadata_path = metadata_path
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # Determine cache file path based on metadata filename
        meta_name = os.path.basename(metadata_path).replace(".csv", "")
        self.cache_file = os.path.join(self.cache_dir, f"{meta_name}_features.npz")

        self.data = None
        self.labels = None
        self.sample_indices = None  # Stores (start, end) in the concatenated arrays
        self.sample_ids = None

        # Load or Compute
        if load_cached_data and os.path.exists(self.cache_file):
            self._load_cache()
        else:
            self._process_and_cache()

        # Prepare indices for access
        if self.mode == "train":
            self.windows = self._make_windows()
        else:
            # For val/test, we access by sample index directly
            pass

    def _process_and_cache(self):
        """Reads raw data, extracts features, and saves to cache."""
        print(f"Processing data from {self.metadata_path}...")
        df = pd.read_csv(self.metadata_path)

        all_features_list = []
        all_labels_list = []
        sample_indices_list = []
        sample_ids_list = []

        current_idx = 0

        # MFCC Transform
        # Standard Kinect audio is usually 16kHz.
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=16000,
            n_mfcc=AUDIO_DIM,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
        )

        for _, row in df.iterrows():
            sample_id = row["sample_id"]
            data_path = os.path.join(INPUT_DIR, row["data_path"])
            audio_path = os.path.join(INPUT_DIR, row["audio_path"])

            # --- Load Skeleton ---
            try:
                mat = scipy.io.loadmat(
                    data_path, squeeze_me=True, struct_as_record=False
                )
                video = mat["Video"]
                num_frames = video.NumFrames
                frames = video.Frames
            except Exception as e:
                # Skip corrupt files or handle gracefully
                continue

            # Extract Skeleton (T, 20, 3)
            skeleton_frames = []

            # Normalize frame access
            if not isinstance(frames, (np.ndarray, list)):
                frames = [frames]

            # Ensure we don't exceed actual frames available
            actual_frames = len(frames)
            if actual_frames != num_frames:
                num_frames = actual_frames

            for i in range(num_frames):
                try:
                    skel = frames[i].Skeleton
                    joints_xyz = np.zeros((20, 3), dtype=np.float32)

                    # Extract joints
                    # Logic to handle Matlab struct array variations
                    if isinstance(skel, (np.ndarray, list)):
                        for j in range(min(len(skel), 20)):
                            wp = skel[j].WorldPosition
                            if hasattr(wp, "X"):
                                joints_xyz[j] = [wp.X, wp.Y, wp.Z]
                            elif isinstance(wp, (np.ndarray, list)) and len(wp) >= 3:
                                joints_xyz[j] = wp[:3]

                    skeleton_frames.append(joints_xyz)
                except Exception:
                    skeleton_frames.append(np.zeros((20, 3), dtype=np.float32))

            skeleton_arr = np.array(skeleton_frames)  # (T, 20, 3)

            if skeleton_arr.shape[0] == 0:
                continue

            # Normalize: Subtract HipCenter (Index 0)
            hip_center = skeleton_arr[:, 0:1, :]  # (T, 1, 3)
            skeleton_norm = skeleton_arr - hip_center

            # Flatten: (T, 60)
            feat_skeleton = skeleton_norm.reshape(num_frames, -1)

            # Velocity: (T, 60)
            feat_velocity = np.zeros_like(feat_skeleton)
            feat_velocity[1:] = feat_skeleton[1:] - feat_skeleton[:-1]

            # --- Load Audio ---
            feat_audio = np.zeros((num_frames, AUDIO_DIM), dtype=np.float32)
            if os.path.exists(audio_path):
                try:
                    waveform, sample_rate = torchaudio.load(audio_path)
                    # Resample to 16k if needed
                    if sample_rate != 16000:
                        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                        waveform = resampler(waveform)

                    # Compute MFCC
                    mfcc = mfcc_transform(waveform)  # (1, n_mfcc, time)
                    mfcc = mfcc.squeeze(0).transpose(0, 1)  # (time, n_mfcc)
                    mfcc_np = mfcc.numpy()

                    # Align to video frames via interpolation
                    if mfcc_np.shape[0] > 0:
                        audio_indices = np.linspace(0, mfcc_np.shape[0] - 1, num_frames)
                        feat_audio_aligned = np.zeros(
                            (num_frames, AUDIO_DIM), dtype=np.float32
                        )
                        for d in range(AUDIO_DIM):
                            feat_audio_aligned[:, d] = np.interp(
                                audio_indices,
                                np.arange(mfcc_np.shape[0]),
                                mfcc_np[:, d],
                            )
                        feat_audio = feat_audio_aligned

                except Exception:
                    pass

            # Concatenate Features
            features = np.concatenate(
                [feat_skeleton, feat_velocity, feat_audio], axis=1
            )

            # --- Labels ---
            labels = np.zeros(num_frames, dtype=np.int64)  # Default 0 (background)

            if self.mode != "test":
                label_list = (
                    json.loads(row["labels"]) if isinstance(row["labels"], str) else []
                )
                for l in label_list:
                    lid = int(l["id"])
                    # Matlab 1-based indexing to 0-based
                    start = int(l["begin"]) - 1
                    end = int(l["end"])

                    start = max(0, start)
                    end = min(num_frames, end)

                    if start < end:
                        labels[start:end] = lid

            # Store
            all_features_list.append(features.astype(np.float32))
            all_labels_list.append(labels.astype(np.int64))
            sample_ids_list.append(str(sample_id))

            length = len(features)
            sample_indices_list.append([current_idx, current_idx + length])
            current_idx += length

        # Concatenate all
        if len(all_features_list) > 0:
            self.data = np.concatenate(all_features_list, axis=0)
            self.labels = np.concatenate(all_labels_list, axis=0)
            self.sample_indices = np.array(sample_indices_list, dtype=np.int64)
            self.sample_ids = np.array(sample_ids_list)
        else:
            self.data = np.zeros(
                (0, SKELETON_DIM + VELOCITY_DIM + AUDIO_DIM), dtype=np.float32
            )
            self.labels = np.zeros((0,), dtype=np.int64)
            self.sample_indices = np.zeros((0, 2), dtype=np.int64)
            self.sample_ids = np.array([])

        self._save_cache()

    def _save_cache(self):
        # Save compressed npz
        np.savez(
            self.cache_file,
            data=self.data,
            labels=self.labels,
            sample_indices=self.sample_indices,
            sample_ids=self.sample_ids,
        )

    def _load_cache(self):
        try:
            loaded = np.load(self.cache_file)
            self.data = loaded["data"]
            self.labels = loaded["labels"]
            self.sample_indices = loaded["sample_indices"]
            self.sample_ids = loaded["sample_ids"]
        except Exception:
            self._process_and_cache()

    def _make_windows(self):
        """Generates sliding window indices for training."""
        window_list = []

        for i in range(len(self.sample_indices)):
            start_global, end_global = self.sample_indices[i]
            sample_len = end_global - start_global

            if sample_len <= WINDOW_SIZE:
                # Sample shorter than window: take the whole thing (will be padded in __getitem__)
                window_list.append((start_global, end_global, True))
            else:
                # Slide window
                curr = start_global
                while curr + WINDOW_SIZE <= end_global:
                    window_list.append((curr, curr + WINDOW_SIZE, False))
                    curr += STRIDE

                # Ensure we cover the end of the sequence
                if curr < end_global:
                    window_list.append((end_global - WINDOW_SIZE, end_global, False))

        return window_list

    def __len__(self):
        if self.mode == "train":
            return len(self.windows)
        else:
            return len(self.sample_indices)

    def __getitem__(self, idx):
        if self.mode == "train":
            start_idx, end_idx, needs_pad = self.windows[idx]

            feat = self.data[start_idx:end_idx]
            lbl = self.labels[start_idx:end_idx]

            if needs_pad:
                pad_len = WINDOW_SIZE - len(feat)
                # Pad features with 0, labels with 0 (background)
                feat = np.pad(feat, ((0, pad_len), (0, 0)), mode="constant")
                lbl = np.pad(lbl, (0, pad_len), mode="constant", constant_values=0)

            return torch.from_numpy(feat), torch.from_numpy(lbl)

        else:
            # Return full sequence
            start_idx, end_idx = self.sample_indices[idx]
            feat = self.data[start_idx:end_idx]
            lbl = self.labels[start_idx:end_idx]
            sid = self.sample_ids[idx]

            return torch.from_numpy(feat), torch.from_numpy(lbl), sid


def get_data_loaders(batch_size=BATCH_SIZE):
    """
    Factory function to create DataLoaders for train, val, and test.
    """
    # Train Set
    train_dataset = GestureDataset(
        TRAIN_METADATA_PATH, mode="train", load_cached_data=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # Val Set
    val_dataset = GestureDataset(VAL_METADATA_PATH, mode="val", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True
    )

    # Test Set
    test_dataset = GestureDataset(
        TEST_METADATA_PATH, mode="test", load_cached_data=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True
    )

    return train_loader, val_loader, test_loader
