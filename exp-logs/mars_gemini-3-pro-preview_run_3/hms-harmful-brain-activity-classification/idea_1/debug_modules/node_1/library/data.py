import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# ==========================================
# Helper Functions
# ==========================================


def load_single_eeg(row, input_dir):
    """
    Loads, slices, downsamples, and normalizes a single EEG sample.
    """
    file_path = os.path.join(input_dir, row["eeg_path"])

    # Read parquet file
    # We only read the required EEG channels
    try:
        df = pd.read_parquet(file_path, columns=Config.EEG_CHANNELS)
        raw_data = df.values  # Shape: (Time, Channels)
    except Exception as e:
        # Fallback for corrupt/missing files (return zeros)
        print(f"Error reading {file_path}: {e}")
        return np.zeros((Config.SEQ_LENGTH, Config.N_CHANNELS), dtype=np.float32)

    # Determine slicing indices
    # Train/Val samples are subsets of consolidated files. Test samples are exact.
    if "eeg_label_offset_seconds" in row and not pd.isna(
        row["eeg_label_offset_seconds"]
    ):
        offset_sec = row["eeg_label_offset_seconds"]
        start_idx = int(offset_sec * Config.ORIGINAL_SAMPLING_RATE)
        end_idx = start_idx + (Config.DURATION * Config.ORIGINAL_SAMPLING_RATE)

        # Handle bounds
        if start_idx < 0:
            start_idx = 0

        segment = raw_data[start_idx:end_idx]

        # Pad if segment is shorter than expected
        expected_len = Config.DURATION * Config.ORIGINAL_SAMPLING_RATE
        if segment.shape[0] < expected_len:
            pad_len = expected_len - segment.shape[0]
            segment = np.pad(segment, ((0, pad_len), (0, 0)), mode="constant")
    else:
        # Test files or files without offset (take full duration or crop)
        segment = raw_data
        expected_len = Config.DURATION * Config.ORIGINAL_SAMPLING_RATE

        if segment.shape[0] > expected_len:
            segment = segment[:expected_len]
        elif segment.shape[0] < expected_len:
            pad_len = expected_len - segment.shape[0]
            segment = np.pad(segment, ((0, pad_len), (0, 0)), mode="constant")

    # Handle NaNs (Replace with 0.0)
    segment = np.nan_to_num(segment, nan=0.0)

    # Downsample (200Hz -> 50Hz)
    # Simple slicing [::4]
    step = int(Config.ORIGINAL_SAMPLING_RATE / Config.TARGET_SAMPLING_RATE)
    segment = segment[::step]

    # Normalize (Standard Scaler per channel)
    # (x - mean) / std
    mean = np.mean(segment, axis=0, keepdims=True)
    std = np.std(segment, axis=0, keepdims=True)
    segment = (segment - mean) / (std + 1e-6)

    return segment.astype(np.float32)


def generate_cache(metadata, input_dir, cache_prefix):
    """
    Processes all samples in metadata and saves to .npy files.
    """
    print(f"Generating cache for {cache_prefix}...")

    num_samples = len(metadata)
    data_shape = (num_samples, Config.SEQ_LENGTH, Config.N_CHANNELS)

    # Pre-allocate memory
    all_data = np.zeros(data_shape, dtype=np.float32)

    # Check if we have targets
    has_targets = all(col in metadata.columns for col in Config.TARGET_COLS)
    if has_targets:
        all_targets = metadata[Config.TARGET_COLS].values.astype(np.float32)
    else:
        all_targets = None

    # Iterate and process
    # Note: iterating rows in pandas can be slow, but necessary here.
    for i, (_, row) in enumerate(metadata.iterrows()):
        all_data[i] = load_single_eeg(row, input_dir)
        if i % 1000 == 0 and i > 0:
            print(f"Processed {i}/{num_samples} samples...")

    # Save to disk
    data_path = os.path.join(Config.WORKING_DIR, f"{cache_prefix}_data.npy")
    np.save(data_path, all_data)

    if has_targets:
        target_path = os.path.join(Config.WORKING_DIR, f"{cache_prefix}_targets.npy")
        np.save(target_path, all_targets)

    print(f"Cache saved to {data_path}")
    return all_data, all_targets


# ==========================================
# Dataset Class
# ==========================================


class EEGDataset(Dataset):
    def __init__(self, metadata, input_dir, data_cache=None, target_cache=None):
        """
        Args:
            metadata (pd.DataFrame): Metadata containing file paths and targets.
            input_dir (str): Base directory for EEG files.
            data_cache (np.ndarray, optional): Pre-loaded data array.
            target_cache (np.ndarray, optional): Pre-loaded target array.
        """
        self.metadata = metadata
        self.input_dir = input_dir
        self.data_cache = data_cache
        self.target_cache = target_cache

        # Check if targets exist in metadata (for on-the-fly loading)
        self.has_targets = all(
            col in self.metadata.columns for col in Config.TARGET_COLS
        )
        if self.has_targets and self.target_cache is None:
            self.targets = self.metadata[Config.TARGET_COLS].values.astype(np.float32)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # 1. Get Input Data
        if self.data_cache is not None:
            # Load from memory
            x = self.data_cache[idx]
        else:
            # Load from disk
            row = self.metadata.iloc[idx]
            x = load_single_eeg(row, self.input_dir)

        # 2. Get Target Data
        if self.target_cache is not None:
            y = self.target_cache[idx]
            return torch.tensor(x, dtype=torch.float32), torch.tensor(
                y, dtype=torch.float32
            )
        elif self.has_targets:
            y = self.targets[idx]
            return torch.tensor(x, dtype=torch.float32), torch.tensor(
                y, dtype=torch.float32
            )
        else:
            # Test set (no targets)
            return torch.tensor(x, dtype=torch.float32)


# ==========================================
# Main Data Loading Function
# ==========================================


def get_dataloaders(load_cached_data=False):
    """
    Prepares DataLoaders for Train, Val, and Test.
    Handles caching logic if requested.
    """
    print("Initializing DataLoaders...")

    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # 2. Handle DEBUG mode
    if Config.DEBUG:
        print(f"DEBUG mode enabled. Subsampling data to {Config.BATCH_SIZE * 2} rows.")
        train_df = train_df.head(Config.BATCH_SIZE * 2)
        val_df = val_df.head(Config.BATCH_SIZE * 2)
        test_df = test_df.head(Config.BATCH_SIZE * 2)

    # 3. Caching Logic
    train_data, train_targets = None, None
    val_data, val_targets = None, None
    # We typically don't cache test data unless specified, but let's keep it simple and only cache train/val
    # to save time during inference if needed, but usually inference is one-pass.

    if load_cached_data:
        suffix = "_debug" if Config.DEBUG else ""

        # --- Train Cache ---
        train_data_path = os.path.join(Config.WORKING_DIR, f"train{suffix}_data.npy")
        train_target_path = os.path.join(
            Config.WORKING_DIR, f"train{suffix}_targets.npy"
        )

        if os.path.exists(train_data_path) and os.path.exists(train_target_path):
            print(f"Loading train cache from {train_data_path}...")
            train_data = np.load(train_data_path)
            train_targets = np.load(train_target_path)
        else:
            train_data, train_targets = generate_cache(
                train_df, Config.INPUT_DIR, f"train{suffix}"
            )

        # --- Val Cache ---
        val_data_path = os.path.join(Config.WORKING_DIR, f"val{suffix}_data.npy")
        val_target_path = os.path.join(Config.WORKING_DIR, f"val{suffix}_targets.npy")

        if os.path.exists(val_data_path) and os.path.exists(val_target_path):
            print(f"Loading val cache from {val_data_path}...")
            val_data = np.load(val_data_path)
            val_targets = np.load(val_target_path)
        else:
            val_data, val_targets = generate_cache(
                val_df, Config.INPUT_DIR, f"val{suffix}"
            )

    # 4. Create Datasets
    train_dataset = EEGDataset(
        train_df, Config.INPUT_DIR, data_cache=train_data, target_cache=train_targets
    )

    val_dataset = EEGDataset(
        val_df, Config.INPUT_DIR, data_cache=val_data, target_cache=val_targets
    )

    test_dataset = EEGDataset(
        test_df,
        Config.INPUT_DIR,
        data_cache=None,  # Usually not cached for inference
        target_cache=None,
    )

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    print(
        f"DataLoaders ready. Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}"
    )
    return train_loader, val_loader, test_loader
