import os
import numpy as np
import pandas as pd
import torch
import cv2
import albumentations as A
from scipy.signal import resample
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class EEGDataset(Dataset):
    def __init__(self, metadata, mode="train", config=Config):
        self.metadata = metadata
        self.mode = mode
        self.config = config

        # Pre-compute paths and IDs
        self.eeg_paths = self.metadata["eeg_path"].values
        self.spec_paths = self.metadata["spectrogram_path"].values
        self.eeg_ids = self.metadata["eeg_id"].values

        # Handle sub_id for unique cache naming
        if "eeg_sub_id" in self.metadata.columns:
            self.sub_ids = self.metadata["eeg_sub_id"].values
        else:
            self.sub_ids = np.arange(len(self.metadata))

        # Load targets and offsets if available
        if self.mode != "test":
            self.targets = self.metadata[self.config.CLASS_NAMES].values.astype(
                np.float32
            )
            self.eeg_offsets = self.metadata["eeg_label_offset_seconds"].values
            self.spec_offsets = self.metadata["spectrogram_label_offset_seconds"].values
        else:
            self.targets = None
            self.eeg_offsets = np.zeros(len(self.metadata))
            self.spec_offsets = np.zeros(len(self.metadata))

        # Augmentations
        # CoarseDropout simulates SpecAugment (masking blocks in time/freq)
        self.spec_transform = (
            A.Compose(
                [
                    A.CoarseDropout(
                        max_holes=8,
                        max_height=32,
                        max_width=32,
                        min_holes=1,
                        min_height=8,
                        min_width=8,
                        fill_value=0,
                        p=0.5,
                    )
                ]
            )
            if mode == "train"
            else None
        )

    def __len__(self):
        return len(self.metadata)

    def _load_eeg(self, path, offset):
        try:
            full_path = os.path.join(self.config.INPUT_DIR, path)
            eeg_df = pd.read_parquet(full_path)

            # Use all available columns (assuming consistent 10-20 system + EKG)
            data = eeg_df.values

            # Calculate indices for 50s crop
            # offset is the start of the subsample in seconds
            start_sample = int(offset * self.config.EEG_SR)
            end_sample = start_sample + int(
                self.config.EEG_DURATION * self.config.EEG_SR
            )

            # Handle boundaries
            total_samples = data.shape[0]
            if self.mode == "test":
                # Test files are exactly 50s
                crop = data[: int(self.config.EEG_DURATION * self.config.EEG_SR), :]
            else:
                if start_sample < 0:
                    start_sample = 0
                crop = data[start_sample:end_sample, :]

            # Pad if necessary
            target_len = int(self.config.EEG_DURATION * self.config.EEG_SR)
            if crop.shape[0] < target_len:
                pad_len = target_len - crop.shape[0]
                crop = np.pad(crop, ((0, pad_len), (0, 0)), mode="constant")
            elif crop.shape[0] > target_len:
                crop = crop[:target_len, :]

            # Fill NaNs
            crop = np.nan_to_num(crop, nan=0.0, posinf=0.0, neginf=0.0)

            # Channel-wise Instance Normalization
            mean = crop.mean(axis=0, keepdims=True)
            std = crop.std(axis=0, keepdims=True) + 1e-6
            crop = (crop - mean) / std

            # Downsample to Target SR (e.g., 200Hz -> 100Hz)
            # resample operates along axis 0 by default
            num_samples = int(self.config.EEG_DURATION * self.config.TARGET_SR)
            crop_resampled = resample(crop, num_samples, axis=0)

            return crop_resampled.astype(np.float32)

        except Exception as e:
            # Return silent zero array on failure
            return np.zeros(
                (
                    int(self.config.EEG_DURATION * self.config.TARGET_SR),
                    self.config.EEG_CHANNELS,
                ),
                dtype=np.float32,
            )

    def _load_spec(self, path, offset):
        try:
            full_path = os.path.join(self.config.INPUT_DIR, path)
            spec_df = pd.read_parquet(full_path)

            # Drop time column if present to keep only frequencies
            if "time" in spec_df.columns:
                spec_df = spec_df.drop(columns=["time"])

            data = spec_df.values

            # Crop 10 minutes (600s)
            # Assuming standard 0.5Hz time resolution (2s per row) for this dataset
            # 600s / 2s = 300 rows
            target_rows = 300

            if self.mode == "test":
                crop = data  # Test files are exactly 10m
            else:
                row_offset = int(offset / 2)
                start_row = row_offset
                end_row = start_row + target_rows

                if start_row < 0:
                    start_row = 0
                crop = data[start_row:end_row, :]

            # Pad if necessary
            if crop.shape[0] < target_rows:
                pad_rows = target_rows - crop.shape[0]
                crop = np.pad(crop, ((0, pad_rows), (0, 0)), mode="constant")
            elif crop.shape[0] > target_rows:
                crop = crop[:target_rows, :]

            # Log Transform (dB scale approximation)
            crop = np.nan_to_num(crop, nan=0.0)
            crop = np.log1p(crop)

            # Resize to fixed resolution (Time, Freq) -> (512, 512)
            # cv2.resize dsize is (width, height) -> (freq, time)
            crop_resized = cv2.resize(
                crop, dsize=self.config.SPEC_SIZE, interpolation=cv2.INTER_LINEAR
            )

            # Standardization
            mean = crop_resized.mean()
            std = crop_resized.std() + 1e-6
            crop_resized = (crop_resized - mean) / std

            return crop_resized.astype(np.float32)

        except Exception as e:
            return np.zeros(self.config.SPEC_SIZE, dtype=np.float32)

    def _get_relative_indices(self, eeg_offset, spec_offset):
        """
        Calculates the time of each spectrogram step relative to the center of the EEG event.
        """
        # Center of the 50s EEG event
        eeg_center = eeg_offset + (self.config.EEG_DURATION / 2.0)

        # Spectrogram window
        spec_start = spec_offset
        spec_end = spec_offset + self.config.SPEC_DURATION

        # Generate absolute time points for the resized spectrogram (Time axis is dim 0)
        # Note: SPEC_SIZE is (512, 512). We assume dim 0 is Time after resize if input was (Time, Freq)
        # However, cv2.resize(src, (W, H)) results in shape (H, W).
        # Our input was (300, Freqs). We resized to (512, 512).
        # So output is (512, 512). Dim 0 is Time.
        spec_times = np.linspace(
            spec_start, spec_end, self.config.SPEC_SIZE[1]
        )  # Height = Time

        # Relative time
        relative_times = spec_times - eeg_center
        return relative_times.astype(np.float32)

    def __getitem__(self, idx):
        eeg_id = self.eeg_ids[idx]
        sub_id = self.sub_ids[idx]

        # 1. Caching Logic
        # We cache the processed numpy arrays to speed up training
        cache_name = f"{eeg_id}_{sub_id}.npz"
        cache_path = os.path.join(self.config.CACHE_DIR, cache_name)

        load_success = False
        data = {}

        # Attempt to load from cache
        if os.path.exists(cache_path):
            try:
                data = np.load(cache_path)
                eeg_data = data["eeg"]
                spec_data = data["spec"]
                rel_indices = data["rel"]
                load_success = True
            except:
                load_success = False

        # If cache miss or fail, process from scratch
        if not load_success:
            eeg_path = self.eeg_paths[idx]
            spec_path = self.spec_paths[idx]
            eeg_offset = self.eeg_offsets[idx]
            spec_offset = self.spec_offsets[idx]

            eeg_data = self._load_eeg(eeg_path, eeg_offset)
            spec_data = self._load_spec(spec_path, spec_offset)
            rel_indices = self._get_relative_indices(eeg_offset, spec_offset)

            # Save to cache (compressed)
            np.savez_compressed(
                cache_path, eeg=eeg_data, spec=spec_data, rel=rel_indices
            )

        # 2. Augmentations (Applied on-the-fly)
        if self.mode == "train":
            # Channel Dropout for EEG
            if np.random.rand() < 0.5:
                num_drop = np.random.randint(1, 4)  # Drop 1 to 3 channels
                channels = eeg_data.shape[1]
                drop_idx = np.random.choice(channels, num_drop, replace=False)
                eeg_data[:, drop_idx] = 0.0

            # SpecAugment for Spectrogram
            if self.spec_transform:
                # Albumentations expects (H, W, C) or (H, W)
                res = self.spec_transform(image=spec_data)["image"]
                spec_data = res

        # 3. Tensor Conversion
        # EEG: (Seq_Len, Channels) -> (Channels, Seq_Len) for 1D CNN
        eeg_tensor = torch.tensor(eeg_data, dtype=torch.float32).permute(1, 0)

        # Spec: (Time, Freq) -> (3, Time, Freq) for EfficientNet (replicate channels)
        spec_tensor = torch.tensor(spec_data, dtype=torch.float32).unsqueeze(0)
        spec_tensor = spec_tensor.repeat(3, 1, 1)

        # Relative Indices
        rel_tensor = torch.tensor(rel_indices, dtype=torch.float32)

        if self.mode != "test":
            target_tensor = torch.tensor(self.targets[idx], dtype=torch.float32)
            return eeg_tensor, spec_tensor, rel_tensor, target_tensor
        else:
            return eeg_tensor, spec_tensor, rel_tensor


def get_dataloaders(
    train_batch_size=Config.BATCH_SIZE,
    val_batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    debug=Config.DEBUG,
    debug_subset_size=Config.DEBUG_SUBSET_SIZE,
):
    """
    Creates DataLoaders for train, validation, and test sets.
    """
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Debug Subsetting
    if debug:
        train_df = train_df.iloc[:debug_subset_size]
        val_df = val_df.iloc[:debug_subset_size]
        # We keep test set intact or small depending on need, but usually full for checking pipeline
        test_df = test_df.iloc[:debug_subset_size]

    # Instantiate Datasets
    train_ds = EEGDataset(train_df, mode="train", config=Config)
    val_ds = EEGDataset(val_df, mode="val", config=Config)
    test_ds = EEGDataset(test_df, mode="test", config=Config)

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
