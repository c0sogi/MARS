import os
import numpy as np
import pandas as pd
import torch
import cv2
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
from library.config import Config

# -----------------------------------------------------------------------------
# Helper Functions for Data Processing
# -----------------------------------------------------------------------------


def _process_row_static(row, config):
    """
    Static helper function to process a single row of data (EEG + Spectrogram).
    Used by both the Dataset class (fallback) and the Caching mechanism.
    """
    # Determine mode based on presence of target columns or specific flags
    # In test.csv, 'seizure_vote' does not exist.
    is_test = "seizure_vote" not in row

    # -------------------------------------------------------------------------
    # 1. Process Spectrogram
    # -------------------------------------------------------------------------
    spec_path = os.path.join(config.INPUT_DIR, row["spectrogram_path"])
    spec_df = pd.read_parquet(spec_path)

    if is_test:
        # Test files are exactly 10 minutes
        spec_data = spec_df.values
    else:
        # Train files are consolidated; extract 10-minute window
        offset = int(row["spectrogram_label_offset_seconds"])

        if "time" in spec_df.columns:
            # If time column exists, use it for precise slicing
            t = spec_df["time"].values
            mask = (t >= offset) & (t < offset + config.SPEC_DURATION_SEC)
            spec_data = spec_df.loc[mask].drop(columns=["time"]).values
        else:
            # Fallback: Assume 0.5Hz resolution (2s per row) common in this dataset
            # 10 minutes = 600s = 300 rows
            start_row = offset // 2
            end_row = start_row + 300
            spec_data = spec_df.iloc[start_row:end_row].values

    # Log Transform (dB scale) and Clipping
    # Clip to avoid -inf or extreme outliers
    spec_data = np.log1p(np.clip(spec_data, np.exp(-4), np.exp(8)))
    spec_data = np.nan_to_num(spec_data, nan=0.0)

    # Resize to Target Dimensions (Height=Time, Width=Freq) -> (512, 512)
    # cv2.resize expects (width, height)
    spec_data = cv2.resize(
        spec_data, config.SPEC_IMG_SIZE, interpolation=cv2.INTER_AREA
    )

    # Standardization (Global-ish or per sample)
    # Per-sample standardization is safer for varying signal strengths
    eps = 1e-6
    spec_mean = spec_data.mean()
    spec_std = spec_data.std()
    spec_data = (spec_data - spec_mean) / (spec_std + eps)

    # -------------------------------------------------------------------------
    # 2. Process EEG
    # -------------------------------------------------------------------------
    eeg_path = os.path.join(config.INPUT_DIR, row["eeg_path"])
    eeg_df = pd.read_parquet(eeg_path)

    if is_test:
        # Test files are exactly 50 seconds
        eeg_vals = eeg_df.values
    else:
        # Train files: extract 50s window
        offset_sec = row["eeg_label_offset_seconds"]
        start_idx = int(offset_sec * config.EEG_RAW_SAMPLE_RATE)
        end_idx = start_idx + config.EEG_DURATION_SEC * config.EEG_RAW_SAMPLE_RATE
        eeg_vals = eeg_df.iloc[start_idx:end_idx].values

    # Downsample 200Hz -> 100Hz
    eeg_vals = eeg_vals[::2, :]

    # Handle NaNs
    eeg_vals = np.nan_to_num(eeg_vals, nan=0.0)

    # Transpose to (Channels, Time)
    eeg_vals = eeg_vals.T

    # Channel-wise Instance Normalization
    mean = eeg_vals.mean(axis=1, keepdims=True)
    std = eeg_vals.std(axis=1, keepdims=True)
    eeg_vals = (eeg_vals - mean) / (std + eps)

    return spec_data, eeg_vals


def cache_dataset_split(df, config, split_name):
    """
    Pre-processes and caches a dataframe split to disk.
    """
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    print(f"Caching {split_name} set ({len(df)} samples)...")

    def process_and_save(row_tuple):
        _, row = row_tuple
        eeg_id = row["eeg_id"]
        # Use sub_id for uniqueness in train, default to 0 for test
        sub_id = int(row.get("eeg_sub_id", 0))

        fname = f"{eeg_id}_{sub_id}.npy"
        fpath = os.path.join(config.CACHE_DIR, fname)

        if os.path.exists(fpath):
            return

        try:
            spec, eeg = _process_row_static(row, config)
            data = {"spec": spec, "eeg": eeg}
            np.save(fpath, data)
        except Exception as e:
            print(f"Error caching {eeg_id}_{sub_id}: {e}")

    # Use ThreadPoolExecutor for IO-bound tasks (reading parquet)
    # Adjust max_workers based on system limits
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(process_and_save, df.iterrows()))
    print(f"Caching {split_name} complete.")


# -----------------------------------------------------------------------------
# Dataset Class
# -----------------------------------------------------------------------------


class BrainActivityDataset(Dataset):
    def __init__(self, df, config, mode="train", augment=False):
        self.df = df
        self.config = config
        self.mode = mode
        self.augment = augment

        # Augmentation pipeline for Spectrograms
        if self.augment:
            self.spec_transform = A.Compose(
                [
                    A.CoarseDropout(
                        max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.5
                    ),
                ]
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        eeg_id = row["eeg_id"]
        sub_id = int(row.get("eeg_sub_id", 0))

        # 1. Load Data (Cache -> Fallback)
        cache_path = os.path.join(self.config.CACHE_DIR, f"{eeg_id}_{sub_id}.npy")

        try:
            data = np.load(cache_path, allow_pickle=True).item()
            spec = data["spec"]
            eeg = data["eeg"]
        except (FileNotFoundError, OSError):
            # Fallback if cache is missing
            spec, eeg = _process_row_static(row, self.config)

        # 2. Augmentations
        if self.augment:
            # Spectrogram Augmentation (Albumentations expects H, W, C usually, or just image)
            # Spec is (H, W).
            augmented = self.spec_transform(image=spec)
            spec = augmented["image"]

            # EEG Augment (Channel Dropout)
            if np.random.rand() < 0.5:
                n_channels = eeg.shape[0]
                # Drop 1 to 4 channels
                num_drop = np.random.randint(1, 5)
                drop_idx = np.random.choice(n_channels, size=num_drop, replace=False)
                eeg[drop_idx, :] = 0.0

        # 3. Format Output
        # Spec: (H, W) -> (3, H, W) for EfficientNet
        spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)

        # EEG: (C, T)
        eeg = torch.tensor(eeg, dtype=torch.float32)

        # Guidance Signal
        # Since we centered the event in the 10-minute window during extraction,
        # the relative position is 0.5.
        guidance = torch.tensor(0.5, dtype=torch.float32)

        # 4. Return
        if self.mode == "test":
            return spec, eeg, guidance
        else:
            # Extract Targets
            # Columns: [seizure_prob, lpd_prob, ...]
            prob_cols = [c.replace("_vote", "_prob") for c in self.config.CLASS_NAMES]
            target = row[prob_cols].values.astype(np.float32)
            target = torch.tensor(target, dtype=torch.float32)

            return spec, eeg, guidance, target


# -----------------------------------------------------------------------------
# Data Loader Factory
# -----------------------------------------------------------------------------


def get_dataloaders(config, load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.
    Handles caching if requested.
    """
    # 1. Load Metadata
    train_df = pd.read_csv(config.TRAIN_CSV)
    val_df = pd.read_csv(config.VAL_CSV)
    test_df = pd.read_csv(config.TEST_CSV)

    # 2. Debug Subsetting
    if config.DEBUG:
        print(
            f"DEBUG Mode: Subsetting data (Train={config.TRAIN_SUBSET_SIZE}, Val={config.VAL_SUBSET_SIZE})"
        )
        train_df = train_df.sample(
            n=min(len(train_df), config.TRAIN_SUBSET_SIZE), random_state=config.SEED
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(len(val_df), config.VAL_SUBSET_SIZE), random_state=config.SEED
        ).reset_index(drop=True)
        # Keep test small in debug
        test_df = test_df.sample(
            n=min(len(test_df), 100), random_state=config.SEED
        ).reset_index(drop=True)

    # 3. Caching
    if load_cached_data:
        cache_dataset_split(train_df, config, "Train")
        cache_dataset_split(val_df, config, "Validation")
        # Cache test data as well for consistency
        cache_dataset_split(test_df, config, "Test")

    # 4. Create Datasets
    train_ds = BrainActivityDataset(train_df, config, mode="train", augment=True)
    val_ds = BrainActivityDataset(val_df, config, mode="val", augment=False)
    test_ds = BrainActivityDataset(test_df, config, mode="test", augment=False)

    # 5. Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
