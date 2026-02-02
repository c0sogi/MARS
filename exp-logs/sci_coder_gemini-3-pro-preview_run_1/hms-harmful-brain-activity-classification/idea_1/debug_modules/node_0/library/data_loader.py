import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from library.config import (
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    TRAIN_SPECS_DIR,
    TEST_SPECS_DIR,
    CACHE_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    TIME_WINDOW,
    FREQ_BINS,
    SPEC_CHANNELS,
    TARGET_COLS,
    SEED,
)

# Fixed time steps for the model input (e.g., 60s window / 0.2s per step = 300 steps)
TARGET_TIME_STEPS = 300
# Frequency bins per channel (Total 400 / 4 channels = 100)
REGION_BINS = FREQ_BINS // SPEC_CHANNELS


def compute_or_load_stats(metadata_df, load_cached_data=True):
    """
    Computes or loads global mean and std for spectrogram normalization.
    Caches the result to avoid re-computation.
    """
    stats_path = os.path.join(CACHE_DIR, "stats.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(stats_path):
        try:
            stats = np.load(stats_path, allow_pickle=True).item()
            return stats["mean"], stats["std"]
        except Exception as e:
            print(f"Failed to load cached stats: {e}. Recomputing...")

    # 2. Compute from scratch
    print("Computing spectrogram statistics from sample...")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Sample up to 500 files for robust estimation
    sample_df = metadata_df.sample(n=min(500, len(metadata_df)), random_state=SEED)

    sum_x = 0
    sum_sq_x = 0
    count = 0

    for _, row in sample_df.iterrows():
        try:
            # Construct full path. Metadata contains relative path (e.g., "train_spectrograms/x.parquet")
            file_path = os.path.join("./input", row["spectrogram_path"])
            if not os.path.exists(file_path):
                continue

            df = pd.read_parquet(file_path)

            # Drop non-feature columns
            if "time" in df.columns:
                df = df.drop(columns=["time"])

            # Convert to numpy and handle NaNs
            data = df.values
            data = np.nan_to_num(data, nan=0.0)

            # Log transform (log1p is safer for 0 values)
            data = np.log1p(data)

            sum_x += np.sum(data)
            sum_sq_x += np.sum(data**2)
            count += data.size

        except Exception:
            continue

    if count == 0:
        # Fallback defaults if reading failed completely
        mean, std = 0.0, 1.0
    else:
        mean = sum_x / count
        std = np.sqrt((sum_sq_x / count) - (mean**2))

    # Avoid division by zero
    if std < 1e-6:
        std = 1.0

    # 3. Save to cache
    np.save(stats_path, {"mean": mean, "std": std})

    print(f"Stats computed: Mean={mean:.4f}, Std={std:.4f}")
    return mean, std


class SpectrogramDataset(Dataset):
    def __init__(self, metadata, mode="train", mean=0.0, std=1.0):
        self.metadata = metadata
        self.mode = mode
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # 1. Load Data
        file_path = os.path.join("./input", row["spectrogram_path"])

        try:
            spec_df = pd.read_parquet(file_path)
        except Exception:
            return self._get_dummy_item()

        # 2. Determine Crop Window
        # Test files are 10 mins (600s), we take the center.
        # Train files are consolidated, we take the labeled offset.
        if self.mode == "test":
            center_sec = 300.0
        else:
            # spectrogram_label_offset_seconds points to the start of the 10s labeled chunk.
            # We want the center of that chunk.
            center_sec = row["spectrogram_label_offset_seconds"] + 5.0

        start_sec = center_sec - (TIME_WINDOW / 2)
        end_sec = center_sec + (TIME_WINDOW / 2)

        # 3. Extract Time Window
        if "time" in spec_df.columns:
            time_vals = spec_df["time"].values
            mask = (time_vals >= start_sec) & (time_vals < end_sec)
            feature_df = spec_df.drop(columns=["time"])
            data = feature_df[mask].values
        else:
            # Fallback if 'time' column is missing
            # Assume rows are linearly distributed over 600 seconds (standard for this dataset's test files)
            total_rows = len(spec_df)
            rows_per_sec = total_rows / 600.0
            start_idx = int(max(0, start_sec * rows_per_sec))
            end_idx = int(min(total_rows, end_sec * rows_per_sec))
            data = spec_df.iloc[start_idx:end_idx].values

        # 4. Handle Empty/Padding
        if data.shape[0] == 0:
            data = np.zeros((TARGET_TIME_STEPS, FREQ_BINS), dtype=np.float32)

        # 5. Resize to Fixed Time Dimension (Interpolation)
        # We need consistent (Time, Freq) shape for batching.
        if data.shape[0] != TARGET_TIME_STEPS:
            # (Batch, Channels, Length) -> (1, Freq, Time)
            tensor_data = torch.tensor(data, dtype=torch.float32).T.unsqueeze(0)
            tensor_data = torch.nn.functional.interpolate(
                tensor_data, size=TARGET_TIME_STEPS, mode="linear", align_corners=False
            )
            data = tensor_data.squeeze(0).T.numpy()  # (Time, Freq)

        # 6. Preprocessing
        data = np.nan_to_num(data, nan=0.0)
        data = np.log1p(data)
        data = (data - self.mean) / self.std

        # 7. Reshape to (Channels, Time, Freq_per_channel)
        # Input shape: (300, 400). Target: (4, 300, 100).
        # We assume columns are ordered by region (LL, RL, LP, RP) or similar blocks.
        try:
            data = data.reshape(TARGET_TIME_STEPS, SPEC_CHANNELS, REGION_BINS)
            data = np.transpose(data, (1, 0, 2))  # -> (4, 300, 100)
        except ValueError:
            data = np.zeros(
                (SPEC_CHANNELS, TARGET_TIME_STEPS, REGION_BINS), dtype=np.float32
            )

        data_tensor = torch.tensor(data, dtype=torch.float32)

        # 8. Return Data and Labels
        if self.mode == "test":
            return data_tensor
        else:
            # Use probability columns if available (soft targets), else votes
            prob_cols = [col.replace("_vote", "_prob") for col in TARGET_COLS]

            if all(c in row.index for c in prob_cols):
                labels = row[prob_cols].values.astype(np.float32)
            else:
                labels = row[TARGET_COLS].values.astype(np.float32)
                if labels.sum() > 0:
                    labels = labels / labels.sum()
                else:
                    labels = np.ones(len(TARGET_COLS)) / len(TARGET_COLS)

            return data_tensor, torch.tensor(labels, dtype=torch.float32)

    def _get_dummy_item(self):
        data = torch.zeros(
            (SPEC_CHANNELS, TARGET_TIME_STEPS, REGION_BINS), dtype=torch.float32
        )
        if self.mode == "test":
            return data
        else:
            labels = torch.ones(len(TARGET_COLS), dtype=torch.float32) / len(
                TARGET_COLS
            )
            return data, labels


def get_dataloaders(
    train_batch_size=BATCH_SIZE,
    val_batch_size=BATCH_SIZE,
    test_batch_size=BATCH_SIZE,
    load_cached_data=True,
):
    """
    Creates and returns DataLoaders for train, val, and test sets.
    """
    print("Initializing DataLoaders...")

    # 1. Load Metadata
    try:
        train_df = pd.read_csv(TRAIN_CSV)
        val_df = pd.read_csv(VAL_CSV)
        test_df = pd.read_csv(TEST_CSV)
    except FileNotFoundError as e:
        print(f"Error loading metadata: {e}")
        return None, None, None

    # 2. Compute/Load Stats (using training data)
    mean, std = compute_or_load_stats(train_df, load_cached_data=load_cached_data)

    # 3. Create Datasets
    train_dataset = SpectrogramDataset(train_df, mode="train", mean=mean, std=std)
    val_dataset = SpectrogramDataset(val_df, mode="val", mean=mean, std=std)
    test_dataset = SpectrogramDataset(test_df, mode="test", mean=mean, std=std)

    # 4. Create DataLoaders
    # Using pin_memory=True for faster transfer to GPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    print(
        f"DataLoaders ready: Train={len(train_loader)} batches, Val={len(val_loader)} batches, Test={len(test_loader)} batches."
    )

    return train_loader, val_loader, test_loader
