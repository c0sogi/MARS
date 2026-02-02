import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import albumentations as A
from library.config import Config

# Standard 10-20 EEG System + EKG (20 Channels)
EEG_CHANNELS = [
    "Fp1",
    "F3",
    "C3",
    "P3",
    "F7",
    "T3",
    "T5",
    "O1",
    "Fz",
    "Cz",
    "Pz",
    "Fp2",
    "F4",
    "C4",
    "P4",
    "F8",
    "T4",
    "T6",
    "O2",
    "EKG",
]

# Spectrogram Regions
SPEC_REGIONS = ["LL", "RL", "LP", "RP"]


class EEGSeizureDataset(Dataset):
    """
    Dataset for Harmful Brain Activity Detection.
    Loads paired EEG and Spectrogram data, applies preprocessing, coordinate injection, and augmentation.
    """

    def __init__(self, df, mode="train", augment=False):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            mode (str): 'train', 'val', or 'test'.
            augment (bool): Whether to apply data augmentation.
        """
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.augment = augment

        # Pre-compute paths
        self.eeg_paths = self.df["eeg_path"].values
        self.spec_paths = self.df["spectrogram_path"].values

        # Handle offsets (Test set defaults to 0)
        if "eeg_label_offset_seconds" in self.df.columns:
            self.eeg_offsets = self.df["eeg_label_offset_seconds"].values
        else:
            self.eeg_offsets = np.zeros(len(self.df))

        if "spectrogram_label_offset_seconds" in self.df.columns:
            self.spec_offsets = self.df["spectrogram_label_offset_seconds"].values
        else:
            self.spec_offsets = np.zeros(len(self.df))

        # Load targets for training/validation
        if self.mode != "test":
            self.targets = self.df[Config.TARGET_COLS].values
        else:
            self.targets = None

        # Augmentation pipeline for Spectrograms (SpecAugment style)
        self.spec_transform = (
            A.Compose(
                [
                    A.XYMasking(
                        num_masks_x=(1, 2),
                        mask_x_length=(10, 40),
                        num_masks_y=(1, 2),
                        mask_y_length=(10, 40),
                        p=0.5,
                    )
                ]
            )
            if augment
            else None
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Load and Process EEG
        eeg_path = os.path.join(Config.INPUT_DIR, self.eeg_paths[idx])
        eeg_offset = self.eeg_offsets[idx]
        eeg_tensor = self.load_eeg(eeg_path, eeg_offset)

        # 2. Load and Process Spectrogram
        spec_path = os.path.join(Config.INPUT_DIR, self.spec_paths[idx])
        spec_offset = self.spec_offsets[idx]
        spec_tensor = self.load_spectrogram(spec_path, spec_offset)

        # 3. Return Data
        if self.mode != "test":
            target = self.targets[idx]
            return eeg_tensor, spec_tensor, torch.tensor(target, dtype=torch.float32)
        else:
            return eeg_tensor, spec_tensor

    def load_eeg(self, path, offset):
        """
        Loads EEG parquet, crops 50s window, downsamples, and normalizes.
        """
        try:
            # Attempt to read specific columns for speed
            eeg_df = pd.read_parquet(path, columns=EEG_CHANNELS)
        except Exception:
            # Fallback for corrupt files or missing columns: return zeros
            return torch.zeros(
                (Config.EEG_CHANNELS, Config.EEG_SEQ_LEN), dtype=torch.float32
            )

        # Calculate indices
        fs = Config.EEG_SR  # 200 Hz
        target_duration = Config.EEG_DURATION  # 50s
        target_len_raw = int(target_duration * fs)  # 10000 samples

        start_sample = int(offset * fs)
        end_sample = start_sample + target_len_raw

        # Extract Data
        if self.mode == "test":
            # Test files are pre-cropped
            data = eeg_df.values
        else:
            # Train files are long recordings
            total_samples = len(eeg_df)
            # Clip indices
            s = max(0, start_sample)
            e = min(total_samples, end_sample)
            data = eeg_df.iloc[s:e].values

        # Pad if shorter than expected
        if len(data) < target_len_raw:
            pad_len = target_len_raw - len(data)
            # Pad at the end
            data = np.pad(data, ((0, pad_len), (0, 0)), mode="constant")
        elif len(data) > target_len_raw:
            data = data[:target_len_raw]

        # Downsample 200Hz -> 100Hz (Decimation)
        data = data[::2, :]  # Shape: (5000, 20)

        # Handle NaNs (replace with 0)
        data = np.nan_to_num(data, nan=0.0)

        # Transpose to (Channels, Time) -> (20, 5000)
        data = data.T

        # Instance Normalization
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        data = (data - mean) / (std + 1e-6)

        # Augmentation: Channel Dropout
        if self.augment and np.random.rand() < Config.EEG_CHANNEL_DROPOUT_PROB:
            # Drop 1 to 3 channels randomly
            num_drop = np.random.randint(1, 4)
            drop_indices = np.random.choice(data.shape[0], num_drop, replace=False)
            data[drop_indices, :] = 0.0

        return torch.tensor(data, dtype=torch.float32)

    def load_spectrogram(self, path, offset):
        """
        Loads Spectrogram parquet, crops 10m window, resizes, adds coordinate map.
        """
        try:
            spec_df = pd.read_parquet(path)
        except Exception:
            return torch.zeros(
                (Config.SPEC_CHANNELS, Config.SPEC_RESIZE_H, Config.SPEC_RESIZE_W),
                dtype=torch.float32,
            )

        # Calculate indices
        # Metadata analysis suggests ~2 seconds per row for spectrograms
        sec_per_row = 2
        window_sec = Config.SPEC_WINDOW  # 600s
        window_rows = int(window_sec / sec_per_row)  # 300 rows

        start_row = int(offset / sec_per_row)
        end_row = start_row + window_rows

        # Extract Data
        if self.mode == "test":
            data = spec_df.values
        else:
            total_rows = len(spec_df)
            s = max(0, start_row)
            e = min(total_rows, end_row)
            data = spec_df.iloc[s:e].values

        # Pad if necessary
        if len(data) < window_rows:
            pad_len = window_rows - len(data)
            data = np.pad(data, ((0, pad_len), (0, 0)), mode="constant")
        elif len(data) > window_rows:
            data = data[:window_rows]

        # Log Transform (handles skewness in power spectra)
        data = np.nan_to_num(data, nan=0.0)
        data = np.log1p(data)

        # Parse Regions and Resize
        # Columns are like 'LL_0.59', 'RL_2.4', etc.
        cols = list(spec_df.columns)
        regions_data = []

        for region in SPEC_REGIONS:
            # Identify columns for this region
            r_cols = [c for c in cols if c.startswith(f"{region}_")]

            if not r_cols:
                # Fallback zero image
                r_img_resized = np.zeros((Config.SPEC_RESIZE_H, Config.SPEC_RESIZE_W))
            else:
                # Extract region matrix
                # Assuming columns are sorted or we rely on pandas order
                r_indices = [cols.index(c) for c in r_cols]
                r_img = data[:, r_indices]

                # Resize to (512, 512)
                # cv2.resize takes (width, height)
                r_img_resized = cv2.resize(
                    r_img,
                    (Config.SPEC_RESIZE_W, Config.SPEC_RESIZE_H),
                    interpolation=cv2.INTER_LINEAR,
                )

            regions_data.append(r_img_resized)

        # Stack Regions -> (4, 512, 512)
        spec_tensor = np.stack(regions_data, axis=0)

        # Normalize (Standardize)
        # Using simple instance standardization for robustness
        mean = spec_tensor.mean()
        std = spec_tensor.std()
        spec_tensor = (spec_tensor - mean) / (std + 1e-6)

        # Augmentation: SpecAugment
        if self.spec_transform:
            # Albumentations expects HWC
            spec_hwc = np.transpose(spec_tensor, (1, 2, 0))
            augmented = self.spec_transform(image=spec_hwc)["image"]
            spec_tensor = np.transpose(augmented, (2, 0, 1))

        # Coordinate Map Injection (5th Channel)
        # Linear gradient from -1 to 1 along the time axis (Height)
        # Encodes "Time from Center"
        time_axis = np.linspace(-1, 1, Config.SPEC_RESIZE_H)
        coord_map = np.tile(time_axis[:, None], (1, Config.SPEC_RESIZE_W))  # (512, 512)
        coord_map = coord_map[np.newaxis, :, :]  # (1, 512, 512)

        # Concatenate -> (5, 512, 512)
        final_spec = np.concatenate([spec_tensor, coord_map], axis=0)

        return torch.tensor(final_spec, dtype=torch.float32)
