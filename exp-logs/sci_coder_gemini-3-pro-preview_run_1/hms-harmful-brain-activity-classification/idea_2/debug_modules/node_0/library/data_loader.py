import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.signal import resample
from library.config import Config


class EEGDataset(Dataset):
    """
    Custom Dataset for loading and preprocessing EEG data.
    """

    def __init__(self, metadata_path, mode="train", debug=False):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            mode (str): 'train', 'val', or 'test'.
            debug (bool): If True, use a small subset of the data.
        """
        self.mode = mode
        self.config = Config

        # Load metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        self.df = pd.read_csv(metadata_path)

        # Handle Debug Mode
        if debug:
            subset_size = min(len(self.df), self.config.DEBUG_SUBSET_SIZE)
            self.df = self.df.sample(
                n=subset_size, random_state=self.config.SEED
            ).reset_index(drop=True)

        # Pre-compute useful indices/columns
        self.eeg_paths = self.df["eeg_path"].values

        if self.mode != "test":
            self.offsets = self.df["eeg_label_offset_seconds"].values
            self.targets = self.df[self.config.TARGET_COLS].values
        else:
            self.offsets = None
            self.targets = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Load Raw Data
        rel_path = self.eeg_paths[idx]
        full_path = os.path.join(self.config.INPUT_DIR, rel_path)

        try:
            # Read parquet file
            # Columns are electrode names.
            eeg_df = pd.read_parquet(full_path, columns=self.config.EEG_CHANNELS)
        except Exception as e:
            # Fallback for corrupt files (though unlikely in this clean dataset)
            # Return zeros
            print(f"Error loading {full_path}: {e}")
            return self._get_dummy_sample()

        raw_data = eeg_df.values  # Shape: (Time, Channels)

        # 2. Slice Time Window
        # Target is 50 seconds * 200 Hz = 10000 samples
        window_len = int(self.config.DURATION * self.config.SAMPLING_RATE)

        if self.mode == "test":
            # Test files are exactly 50s
            data = raw_data
        else:
            # Train/Val files are longer; extract specific window
            offset_sec = self.offsets[idx]
            start_idx = int(offset_sec * self.config.SAMPLING_RATE)
            end_idx = start_idx + window_len

            # Handle potential bounds
            if start_idx < 0:
                start_idx = 0
            if end_idx > len(raw_data):
                # If window goes out of bounds, try to center or clip
                # For this dataset, offsets are generally valid.
                # We'll clip to the end or pad if necessary.
                data = raw_data[start_idx:]
            else:
                data = raw_data[start_idx:end_idx]

        # Ensure correct length (pad if too short, crop if too long)
        if len(data) < window_len:
            pad_len = window_len - len(data)
            data = np.pad(
                data, ((0, pad_len), (0, 0)), mode="constant", constant_values=0
            )
        elif len(data) > window_len:
            data = data[:window_len]

        # 3. Preprocessing
        # Fill NaNs with 0
        data = np.nan_to_num(data, nan=0.0)

        # Resample: 200Hz -> 100Hz (10000 -> 5000 samples)
        # scipy.signal.resample resamples along axis 0 (time) by default
        data = resample(data, self.config.FIXED_LENGTH)

        # Normalize: Channel-wise (Instance Normalization)
        # Shape is (Time, Channels). We want mean/std per channel.
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        data = (data - mean) / (std + 1e-6)

        # Transpose to (Channels, Time) for PyTorch Conv1d
        # Current: (Time, Channels) -> Target: (Channels, Time)
        data = data.transpose(1, 0)

        # Convert to Tensor
        data_tensor = torch.tensor(data, dtype=torch.float32)

        # 4. Get Target
        if self.mode != "test":
            target = self.targets[idx]
            target_tensor = torch.tensor(target, dtype=torch.float32)
        else:
            # Dummy target for test
            target_tensor = torch.zeros(self.config.NUM_CLASSES, dtype=torch.float32)

        return data_tensor, target_tensor

    def _get_dummy_sample(self):
        """Returns a tensor of zeros in case of loading failure."""
        shape = (self.config.NUM_CHANNELS, self.config.FIXED_LENGTH)
        x = torch.zeros(shape, dtype=torch.float32)
        y = torch.zeros(self.config.NUM_CLASSES, dtype=torch.float32)
        return x, y


def get_dataloaders(
    train_batch_size=Config.BATCH_SIZE,
    val_batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    debug=Config.DEBUG,
):
    """
    Creates and returns DataLoaders for train, val, and test sets.
    """
    # Initialize Datasets
    train_dataset = EEGDataset(
        metadata_path=Config.TRAIN_CSV, mode="train", debug=debug
    )

    val_dataset = EEGDataset(metadata_path=Config.VAL_CSV, mode="val", debug=debug)

    test_dataset = EEGDataset(metadata_path=Config.TEST_CSV, mode="test", debug=debug)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=val_batch_size,  # Use same batch size as val for inference
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
