import os
import numpy as np
import pandas as pd
import torch
import cv2
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from library.config import Config


def load_data_splits(config, load_cached_data=True):
    """
    Loads train, validation, and test metadata.
    Implements caching for the processed metadata dataframes to satisfy the
    requirement for deterministic data processing caching.
    """
    cache_dir = config.working_dir
    os.makedirs(cache_dir, exist_ok=True)

    splits = {}
    for mode in ["train", "val", "test"]:
        cache_path = os.path.join(cache_dir, f"{mode}_meta_cached.parquet")

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached {mode} metadata from {cache_path}")
            df = pd.read_parquet(cache_path)
        else:
            # Determine source file
            if mode == "train":
                csv_path = config.train_csv
            elif mode == "val":
                csv_path = config.val_csv
            else:
                csv_path = config.test_csv

            if not os.path.exists(csv_path):
                print(f"Warning: {csv_path} not found. Skipping {mode}.")
                splits[mode] = None
                continue

            df = pd.read_csv(csv_path)

            # Ensure path columns are strings
            if "eeg_path" in df.columns:
                df["eeg_path"] = df["eeg_path"].astype(str)
            if "spectrogram_path" in df.columns:
                df["spectrogram_path"] = df["spectrogram_path"].astype(str)

            # Save to cache for future runs
            print(f"Saving {mode} metadata to cache at {cache_path}")
            df.to_parquet(cache_path, index=False)

        splits[mode] = df

    return splits["train"], splits["val"], splits["test"]


class EEGTransform:
    """
    Augmentation pipeline for 1D EEG signals.
    Includes Amplitude Scaling and Time Masking.
    """

    def __init__(self, mode="train", time_mask_prob=0.3, amplitude_scale_prob=0.5):
        self.mode = mode
        self.time_mask_prob = time_mask_prob
        self.amplitude_scale_prob = amplitude_scale_prob

    def __call__(self, x):
        # Input x shape: (Time, Channels)
        if self.mode != "train":
            return x

        # Amplitude Scaling
        if np.random.rand() < self.amplitude_scale_prob:
            scale = np.random.uniform(0.8, 1.2)
            x = x * scale

        # Time Masking
        if np.random.rand() < self.time_mask_prob:
            T, C = x.shape
            # Mask between 5% and 20% of the signal
            mask_len = np.random.randint(T // 20, T // 5)
            start = np.random.randint(0, T - mask_len)
            x[start : start + mask_len, :] = 0

        return x


def get_spec_transforms(mode="train"):
    """
    Augmentation pipeline for 2D Spectrograms using Albumentations.
    Includes CoarseDropout to simulate SpecAugment.
    """
    if mode == "train":
        return A.Compose(
            [
                A.CoarseDropout(
                    max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.5
                ),
            ]
        )
    return A.Compose([])


class EEGSpecDataset(Dataset):
    """
    Dual-Stream Dataset for EEG and Spectrograms.
    Loads raw EEG (1D) and Spectrograms (2D) from parquet files.
    """

    def __init__(self, df, config, mode="train"):
        self.df = df
        self.config = config
        self.mode = mode

        self.eeg_transform = EEGTransform(mode)
        self.spec_transform = get_spec_transforms(mode)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ---------------------------------------------------------------------
        # 1. Process Raw EEG
        # ---------------------------------------------------------------------
        eeg_path = os.path.join(self.config.input_dir, row["eeg_path"])
        offset_sec = row.get("eeg_label_offset_seconds", 0)

        # Initialize container for EEG data
        target_len = self.config.eeg_seq_len  # 5000 samples (50s * 100Hz)
        num_channels = len(self.config.channel_names)
        eeg_data = np.zeros((target_len, num_channels), dtype=np.float32)

        try:
            # Read specific channels from parquet
            full_eeg = pd.read_parquet(
                eeg_path, columns=self.config.channel_names
            ).values

            # Calculate indices based on original sampling rate (200Hz)
            fs_orig = self.config.eeg_sr_original
            start_idx = int(offset_sec * fs_orig)
            duration_samples = int(self.config.eeg_duration * fs_orig)  # 10000 samples
            end_idx = start_idx + duration_samples

            # Handle boundary conditions
            total_samples = full_eeg.shape[0]
            if start_idx >= total_samples:
                start_idx = max(0, total_samples - duration_samples)
                end_idx = total_samples

            raw_crop = full_eeg[start_idx:end_idx]

            # Pad if crop is shorter than expected duration
            if raw_crop.shape[0] < duration_samples:
                pad = duration_samples - raw_crop.shape[0]
                raw_crop = np.pad(raw_crop, ((0, pad), (0, 0)), mode="constant")

            # Downsample from 200Hz to 100Hz
            downsampled = raw_crop[::2, :]

            # Ensure exact length matches target
            if downsampled.shape[0] > target_len:
                downsampled = downsampled[:target_len]
            elif downsampled.shape[0] < target_len:
                pad = target_len - downsampled.shape[0]
                downsampled = np.pad(downsampled, ((0, pad), (0, 0)), mode="constant")

            eeg_data = downsampled

        except Exception as e:
            # Fallback to zeros on error
            pass

        # Handle NaNs and Normalize
        eeg_data = np.nan_to_num(eeg_data, nan=0.0)

        # Channel-wise Instance Normalization
        mean = np.mean(eeg_data, axis=0, keepdims=True)
        std = np.std(eeg_data, axis=0, keepdims=True)
        eeg_data = (eeg_data - mean) / (std + 1e-6)

        # Apply Augmentation
        eeg_data = self.eeg_transform(eeg_data)

        # Convert to Tensor: (Time, Channels) -> (Channels, Time)
        eeg_tensor = torch.tensor(eeg_data, dtype=torch.float32).permute(1, 0)

        # ---------------------------------------------------------------------
        # 2. Process Spectrogram
        # ---------------------------------------------------------------------
        spec_path = os.path.join(self.config.input_dir, row["spectrogram_path"])
        spec_offset_sec = row.get("spectogram_label_offset_seconds", 0)

        # Initialize container for Spectrogram
        spec_h, spec_w = self.config.spec_size
        spec_img = np.zeros((spec_h, spec_w), dtype=np.float32)

        try:
            spec_df = pd.read_parquet(spec_path)
            # Remove 'time' column if it exists
            if "time" in spec_df.columns:
                spec_df = spec_df.drop(columns=["time"])

            spec_arr = spec_df.values

            # Slice 10 minutes (600s)
            # Heuristic: 1 row = 2 seconds (0.5 Hz) -> 300 rows
            rows_per_sec = 0.5
            window_rows = 300
            start_row = int(spec_offset_sec * rows_per_sec)
            end_row = start_row + window_rows

            # Handle boundary conditions
            if start_row >= spec_arr.shape[0]:
                start_row = max(0, spec_arr.shape[0] - window_rows)
                end_row = spec_arr.shape[0]

            spec_crop = spec_arr[start_row:end_row, :]

            # Pad if crop is shorter than expected
            if spec_crop.shape[0] < window_rows:
                pad = window_rows - spec_crop.shape[0]
                spec_crop = np.pad(spec_crop, ((0, pad), (0, 0)), mode="constant")

            # Log Transform
            spec_crop = np.log1p(np.abs(spec_crop))

            # Handle NaNs
            spec_crop = np.nan_to_num(spec_crop, nan=0.0)

            # Resize to target dimensions (512x512)
            spec_img = cv2.resize(spec_crop, (spec_w, spec_h))

            # Min-Max Normalization to [0, 1]
            s_min = spec_img.min()
            s_max = spec_img.max()
            if s_max - s_min > 1e-6:
                spec_img = (spec_img - s_min) / (s_max - s_min)
            else:
                spec_img = np.zeros_like(spec_img)

        except Exception as e:
            pass

        # Apply Augmentation (Requires H, W, C format)
        if self.mode == "train":
            spec_img = spec_img[..., np.newaxis]
            augmented = self.spec_transform(image=spec_img)["image"]
            spec_img = augmented[..., 0]

        # Convert to Tensor: (H, W) -> (1, H, W)
        spec_tensor = torch.tensor(spec_img, dtype=torch.float32).unsqueeze(0)

        # ---------------------------------------------------------------------
        # 3. Return Data and Targets
        # ---------------------------------------------------------------------
        if self.mode != "test":
            # Load probability targets
            targets = row[self.config.prob_cols].values.astype(np.float32)
            return eeg_tensor, spec_tensor, torch.tensor(targets)
        else:
            return eeg_tensor, spec_tensor


def get_dataloaders(config, load_cached_data=True):
    """
    Factory function to create DataLoaders for train, val, and test sets.
    """
    df_train, df_val, df_test = load_data_splits(
        config, load_cached_data=load_cached_data
    )

    loaders = {}

    # Train Loader
    if df_train is not None:
        if config.debug:
            df_train = df_train.iloc[: config.debug_subset_size]

        train_ds = EEGSpecDataset(df_train, config, mode="train")
        loaders["train"] = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    # Validation Loader
    if df_val is not None:
        if config.debug:
            df_val = df_val.iloc[: config.debug_subset_size]

        val_ds = EEGSpecDataset(df_val, config, mode="val")
        loaders["val"] = DataLoader(
            val_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )

    # Test Loader
    if df_test is not None:
        test_ds = EEGSpecDataset(df_test, config, mode="test")
        loaders["test"] = DataLoader(
            test_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )

    return loaders
