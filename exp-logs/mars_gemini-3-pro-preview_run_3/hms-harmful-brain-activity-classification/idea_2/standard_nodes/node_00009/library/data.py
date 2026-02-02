import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from library import config
from library import utils

# Ensure reproducibility
utils.seed_everything(config.SEED)


class EEGDataset(Dataset):
    """
    Dataset class for loading EEG data, downsampling, and preparing for 1D CNN input.
    Cite solution_lesson_node_00006: Avoiding channel stacking/resizing by using raw signals.
    """

    def __init__(self, df, mode="train", augment=False):
        self.df = df
        self.mode = mode
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Resolve File Path
        file_path = os.path.join(config.INPUT_DIR, row["eeg_path"])

        # 2. Load EEG Signal
        try:
            eeg_df = pd.read_parquet(file_path, columns=config.EEG_CHANNELS)
            raw_data = eeg_df.values  # Shape: (Total_Time, 19)
        except Exception as e:
            raw_data = np.zeros(
                (config.TOTAL_SAMPLES, config.N_CHANNELS), dtype=np.float32
            )

        # 3. Extract 50s Window
        if self.mode == "test":
            start_idx = 0
        else:
            offset_sec = row.get("eeg_label_offset_seconds", 0)
            start_idx = int(offset_sec * config.SAMPLING_RATE)

        end_idx = start_idx + config.TOTAL_SAMPLES
        total_available = raw_data.shape[0]

        if start_idx < 0:
            start_idx = 0
            end_idx = config.TOTAL_SAMPLES

        if end_idx <= total_available:
            window = raw_data[start_idx:end_idx, :]
        else:
            window = raw_data[start_idx:, :]
            pad_len = config.TOTAL_SAMPLES - window.shape[0]
            if pad_len > 0:
                padding = np.zeros((pad_len, config.N_CHANNELS), dtype=window.dtype)
                window = np.concatenate([window, padding], axis=0)
            else:
                window = np.zeros(
                    (config.TOTAL_SAMPLES, config.N_CHANNELS), dtype=np.float32
                )

        # 4. Handle NaNs
        for c in range(config.N_CHANNELS):
            col = window[:, c]
            if np.isnan(col).any():
                mean_val = np.nanmean(col)
                if np.isnan(mean_val):
                    mean_val = 0.0
                col[np.isnan(col)] = mean_val
                window[:, c] = col

        # 5. Downsample
        # Cite solution_lesson_node_00001: Downsample to 50Hz (factor 4)
        window = window[:: config.DOWNSAMPLE_FACTOR, :]

        # 6. Clip and Normalize
        window = np.clip(window, -1024, 1024)
        # Robust scaling approximation
        window = window / 32.0

        # 7. Transpose to (Channels, Time) for 1D CNN
        tensor_eeg = torch.tensor(window, dtype=torch.float32).t()

        # 8. Return Data and Target
        if self.mode != "test":
            target = row[config.TARGET_COLS].values.astype(np.float32)
            return tensor_eeg, torch.tensor(target)
        else:
            return tensor_eeg, torch.tensor(0.0)


def get_dataloaders(
    train_batch_size=config.BATCH_SIZE,
    val_batch_size=config.BATCH_SIZE,
    debug=config.DEBUG,
    load_cached_data=False,
):
    """
    Constructs DataLoaders for train, validation, and test sets.

    Args:
        train_batch_size (int): Batch size for training.
        val_batch_size (int): Batch size for validation/inference.
        debug (bool): If True, subsets the data for rapid testing.
        load_cached_data (bool): Placeholder for caching logic.
                                 Due to the large size of spectrogram data (80GB+),
                                 we perform on-the-fly processing rather than disk caching.

    Returns:
        train_loader, val_loader, test_loader
    """

    # Load Metadata
    train_df = pd.read_csv(config.TRAIN_CSV)
    val_df = pd.read_csv(config.VAL_CSV)
    test_df = pd.read_csv(config.TEST_CSV)

    # Debug Subsampling
    if debug:
        print(f"Debug Mode: Sampling {config.DEBUG_SIZE} rows per split.")
        train_df = train_df.head(config.DEBUG_SIZE)
        val_df = val_df.head(config.DEBUG_SIZE)
        test_df = test_df.head(config.DEBUG_SIZE)

    # Instantiate Datasets
    # Augmentation is enabled only for training
    train_dataset = EEGDataset(train_df, mode="train", augment=True)
    val_dataset = EEGDataset(val_df, mode="val", augment=False)
    test_dataset = EEGDataset(test_df, mode="test", augment=False)

    # Instantiate DataLoaders
    # num_workers > 0 ensures data processing happens in parallel processes
    # pin_memory=True speeds up transfer to GPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
