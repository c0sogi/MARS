import os
import numpy as np
import pandas as pd
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Optional, List

from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger("data")


def load_eeg(
    path: str, offset_seconds: float, duration: int = 50, target_sr: int = 100
) -> np.ndarray:
    """
    Loads, slices, downsamples, and normalizes EEG data.

    Args:
        path: Path to the parquet file.
        offset_seconds: Start time of the 50s window.
        duration: Duration in seconds to load (default 50).
        target_sr: Target sampling rate (default 100Hz).

    Returns:
        np.ndarray: Processed EEG data of shape (Channels, Time).
    """
    try:
        # Load parquet
        df = pd.read_parquet(path)

        # Calculate indices (Raw data is 200Hz)
        raw_sr = 200
        start_idx = int(offset_seconds * raw_sr)
        end_idx = start_idx + (duration * raw_sr)

        # Handle slicing
        if "time" in df.columns:
            # If time column exists, use it for more precise alignment, though index is usually reliable
            # For this dataset, usually we rely on index for EEG
            pass

        # Select columns (19 EEG + 1 EKG)
        # Standard 10-20 system + EKG
        feature_cols = [
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

        # Check if columns exist, if not, try to find them or pad
        available_cols = [c for c in feature_cols if c in df.columns]

        # Extract data
        # Handle edge cases where offset is out of bounds or file is short
        file_len = len(df)

        if start_idx < 0:
            start_idx = 0

        # Pad if window goes beyond file
        if end_idx > file_len:
            # Read what we can
            data = df[available_cols].iloc[start_idx:file_len].values
            # Pad the rest with zeros
            pad_len = end_idx - file_len
            if pad_len > 0:
                data = np.pad(data, ((0, pad_len), (0, 0)), mode="constant")
        else:
            data = df[available_cols].iloc[start_idx:end_idx].values

        # Handle missing columns (fill with 0)
        if len(available_cols) < len(feature_cols):
            full_data = np.zeros((len(data), len(feature_cols)), dtype=np.float32)
            for i, col in enumerate(feature_cols):
                if col in available_cols:
                    src_idx = available_cols.index(col)
                    full_data[:, i] = data[:, src_idx]
            data = full_data

        # Fill NaNs
        data = np.nan_to_num(data, nan=0.0)

        # Downsample (200Hz -> 100Hz)
        # Simple decimation is usually sufficient for this task
        step = raw_sr // target_sr
        data = data[::step, :]

        # Clip outliers (simple artifact removal)
        data = np.clip(data, -1024, 1024)

        # Instance Normalization (Channel-wise)
        # (Time, Chan) -> (Chan, Time) for PyTorch
        data = data.transpose(1, 0)

        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        data = (data - mean) / (std + 1e-6)

        return data.astype(np.float32)

    except Exception as e:
        logger.error(f"Error loading EEG {path}: {e}")
        # Return zero tensor of correct shape
        return np.zeros((20, duration * target_sr), dtype=np.float32)


def load_spectrogram(path: str, offset_seconds: float) -> np.ndarray:
    """
    Loads spectrogram, slices 10 min window, reshapes to 4 regions,
    resizes to 512x512, and adds coordinate map.

    Args:
        path: Path to parquet file.
        offset_seconds: Start time of the 10-minute window.

    Returns:
        np.ndarray: Spectrogram tensor of shape (5, 512, 512).
    """
    try:
        df = pd.read_parquet(path)

        # Determine window
        # Spectrogram rows usually correspond to 2 seconds (0.5 Hz) or similar in this dataset
        # But we should rely on 'time' column if it exists

        start_time = offset_seconds
        end_time = offset_seconds + 600  # 10 minutes

        if "time" in df.columns:
            # Filter by time
            # Handle cases where window is outside
            mask = (df["time"] >= start_time) & (df["time"] < end_time)
            sub_df = df.loc[mask]

            # If empty or short, we might need to pad or take nearest
            if len(sub_df) == 0:
                # Fallback: take centered slice based on index if time fails?
                # Or just return zeros
                return np.zeros(
                    (5, Config.SPEC_HEIGHT, Config.SPEC_WIDTH), dtype=np.float32
                )
        else:
            # Fallback if no time column (unlikely for spec files)
            # Assume 2s per row?
            # Let's just return zeros to be safe if format is unexpected
            return np.zeros(
                (5, Config.SPEC_HEIGHT, Config.SPEC_WIDTH), dtype=np.float32
            )

        # Extract 4 regions
        # Columns are like "LL_0.59", "RL_0.59", etc.
        regions = ["LL", "RL", "LP", "RP"]
        region_maps = []

        for region in regions:
            # Select columns starting with region_
            cols = [c for c in sub_df.columns if c.startswith(f"{region}_")]
            if not cols:
                # Should not happen
                region_data = np.zeros((len(sub_df), 100))  # dummy
            else:
                region_data = sub_df[cols].values

            # Log transform
            region_data = np.clip(region_data, np.exp(-4), np.exp(8))
            region_data = np.log(region_data)

            # Handle NaNs
            region_data = np.nan_to_num(region_data, nan=0.0)

            # Resize to (512, 512)
            # Input shape: (Time_Steps, Freq_Bins)
            # Resize expects (Width, Height) -> (Time, Freq)
            # We want output (512, 512)
            resized = cv2.resize(
                region_data,
                (Config.SPEC_WIDTH, Config.SPEC_HEIGHT),
                interpolation=cv2.INTER_LINEAR,
            )

            # Standardize (Global stats approximation or instance)
            # Instance norm for spectrogram image
            mean = resized.mean()
            std = resized.std()
            if std > 1e-6:
                resized = (resized - mean) / std
            else:
                resized = resized - mean

            region_maps.append(resized)

        # Stack 4 regions -> (4, 512, 512)
        # region_maps are (512, 512)
        spec_tensor = np.stack(region_maps, axis=0)

        # Generate Coordinate Map (5th channel)
        # Encodes distance from center (time axis)
        # Shape (512, 512). Varies along Width (Time).
        # Center of window is at x = 256 (0.5)
        x = np.linspace(-1, 1, Config.SPEC_WIDTH)
        # Gaussian focused on center
        coord_vec = np.exp(
            -0.5 * (x**2) / (0.5**2)
        )  # Sigma=0.5 covers most of the window
        # Broadcast to (Height, Width)
        coord_map = np.tile(coord_vec, (Config.SPEC_HEIGHT, 1))

        # Add to stack
        coord_map = coord_map[np.newaxis, :, :]  # (1, 512, 512)
        final_tensor = np.concatenate([spec_tensor, coord_map], axis=0)

        return final_tensor.astype(np.float32)

    except Exception as e:
        logger.error(f"Error loading Spectrogram {path}: {e}")
        return np.zeros((5, Config.SPEC_HEIGHT, Config.SPEC_WIDTH), dtype=np.float32)


class EEGMultiModalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, config: Config, mode: str = "train"):
        self.df = df
        self.config = config
        self.mode = mode
        self.eeg_dir = config.TRAIN_EEGS_DIR if mode != "test" else config.TEST_EEGS_DIR
        self.spec_dir = (
            config.TRAIN_SPECS_DIR if mode != "test" else config.TEST_SPECS_DIR
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load EEG
        # For test, offset is 0
        eeg_offset = row["eeg_label_offset_seconds"] if self.mode != "test" else 0
        eeg_path = os.path.join(Config.INPUT_DIR, row["eeg_path"])

        eeg_data = load_eeg(
            eeg_path,
            eeg_offset,
            duration=Config.EEG_DURATION_S,
            target_sr=Config.EEG_TARGET_SR,
        )

        # 2. Load Spectrogram
        spec_offset = (
            row["spectrogram_label_offset_seconds"] if self.mode != "test" else 0
        )
        spec_path = os.path.join(Config.INPUT_DIR, row["spectrogram_path"])

        spec_data = load_spectrogram(spec_path, spec_offset)

        # 3. Augmentation (Train only)
        if self.mode == "train":
            # Channel Dropout on EEG
            if np.random.random() < 0.5:
                # Drop 1-3 channels
                num_drop = np.random.randint(1, 4)
                drop_idx = np.random.choice(eeg_data.shape[0], num_drop, replace=False)
                eeg_data[drop_idx, :] = 0.0

            # SpecAugment on Spectrogram (Channels 0-3 only)
            # Time Masking
            if np.random.random() < 0.5:
                time_mask_width = np.random.randint(10, 50)
                t0 = np.random.randint(0, Config.SPEC_WIDTH - time_mask_width)
                spec_data[:4, :, t0 : t0 + time_mask_width] = 0.0

            # Freq Masking
            if np.random.random() < 0.5:
                freq_mask_height = np.random.randint(10, 50)
                f0 = np.random.randint(0, Config.SPEC_HEIGHT - freq_mask_height)
                spec_data[:4, f0 : f0 + freq_mask_height, :] = 0.0

        # 4. Prepare Output
        output = {
            "eeg": torch.tensor(eeg_data, dtype=torch.float32),
            "spec": torch.tensor(spec_data, dtype=torch.float32),
        }

        if self.mode != "test":
            # Get probabilities
            probs = row[
                [c for c in self.df.columns if c.endswith("_prob")]
            ].values.astype(np.float32)
            output["target"] = torch.tensor(probs, dtype=torch.float32)

        return output


def get_dataloaders(
    debug: bool = False, batch_size: int = Config.BATCH_SIZE
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates DataLoaders for train, val, and test sets.
    Implements Global Random Subsampling for the training set.
    """
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Debug mode: shrink datasets
    if debug:
        train_df = train_df.head(100)
        val_df = val_df.head(100)
        test_df = test_df.head(100)
        logger.info("Debug mode: Datasets truncated to 100 samples.")
    else:
        # Global Random Subsampling for Training
        # To ensure epoch consistency and fit within time limits
        if len(train_df) > Config.TRAIN_SAMPLE_SIZE:
            train_df = train_df.sample(
                n=Config.TRAIN_SAMPLE_SIZE, random_state=Config.SEED
            ).reset_index(drop=True)
            logger.info(f"Subsampled training set to {len(train_df)} samples.")

    # Create Datasets
    train_ds = EEGMultiModalDataset(train_df, Config, mode="train")
    val_ds = EEGMultiModalDataset(val_df, Config, mode="val")
    test_ds = EEGMultiModalDataset(test_df, Config, mode="test")

    # Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
