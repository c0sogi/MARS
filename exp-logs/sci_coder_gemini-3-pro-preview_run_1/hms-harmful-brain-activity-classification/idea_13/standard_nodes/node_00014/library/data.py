import os
import torch
import pandas as pd
import numpy as np
import scipy.signal
import cv2
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# --- Constants ---
# Standard 10-20 EEG Montage + EKG
EEG_COLUMNS = [
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

# Spectrogram regions in the parquet files
SPEC_REGION_PREFIXES = ["LL", "RL", "LP", "RP"]


class MultiModalDataset(Dataset):
    def __init__(self, metadata, mode="train", augment=False):
        """
        Args:
            metadata (pd.DataFrame): DataFrame containing file paths and labels.
            mode (str): 'train', 'val', or 'test'.
            augment (bool): Whether to apply data augmentation.
        """
        self.metadata = metadata
        self.mode = mode
        self.augment = augment

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # 1. Load and Process EEG
        eeg_tensor = self._load_eeg(row)

        # 2. Load and Process Spectrogram
        spec_tensor = self._load_spectrogram(row)

        # 3. Return Data
        if self.mode != "test":
            # Load targets
            label = row[Config.TARGET_COLS].values.astype(np.float32)
            return eeg_tensor, spec_tensor, torch.tensor(label)
        else:
            return eeg_tensor, spec_tensor

    def _load_eeg(self, row):
        """
        Loads, slices, resamples, and normalizes raw EEG data.
        """
        eeg_path = os.path.join(Config.INPUT_DIR, row["eeg_path"])

        # Read Parquet
        try:
            # Attempt to read only necessary columns for speed
            eeg_df = pd.read_parquet(eeg_path, columns=EEG_COLUMNS)
        except Exception:
            # Fallback: read all and filter (handles missing columns by filling 0 later)
            try:
                eeg_df = pd.read_parquet(eeg_path)
            except Exception:
                # Critical failure fallback: return zeros
                return torch.zeros(
                    (Config.EEG_CHANNELS, Config.EEG_SEQ_LEN), dtype=torch.float32
                )

        # Ensure all required columns exist
        for col in EEG_COLUMNS:
            if col not in eeg_df.columns:
                eeg_df[col] = 0.0
        eeg_df = eeg_df[EEG_COLUMNS]

        # Calculate Time Slice Indices
        # Test files are exactly 50s (offset 0), Train/Val use offsets
        if self.mode == "test":
            start_idx = 0
        else:
            offset_sec = row["eeg_label_offset_seconds"]
            start_idx = int(offset_sec * Config.EEG_ORIGINAL_SR)

        target_duration_samples = Config.EEG_DURATION * Config.EEG_ORIGINAL_SR  # 10000
        end_idx = start_idx + target_duration_samples

        # Extract numpy array
        full_data = eeg_df.values
        total_samples = full_data.shape[0]

        # Handle Boundary Conditions (Padding/Truncation)
        if start_idx < 0:
            start_idx = 0

        # Slice
        # If end_idx exceeds data length, we slice up to end and pad later
        actual_end = min(end_idx, total_samples)
        data = full_data[start_idx:actual_end, :]

        # Pad if necessary
        if data.shape[0] < target_duration_samples:
            pad_len = target_duration_samples - data.shape[0]
            # Pad with zeros at the end
            data = np.pad(
                data, ((0, pad_len), (0, 0)), mode="constant", constant_values=0
            )

        # Handle NaNs (replace with 0)
        data = np.nan_to_num(data, nan=0.0)

        # Resample: 200Hz -> 100Hz
        # Input: (10000, 20) -> Output: (5000, 20)
        data_resampled = scipy.signal.resample(data, Config.EEG_SEQ_LEN, axis=0)

        # Normalize (Channel-wise Standardization)
        # (x - mean) / (std + eps)
        mean = np.mean(data_resampled, axis=0, keepdims=True)
        std = np.std(data_resampled, axis=0, keepdims=True)
        data_norm = (data_resampled - mean) / (std + 1e-6)

        # Clip outliers
        data_norm = np.clip(data_norm, -20, 20)

        # Transpose to (Channels, Time): (20, 5000)
        data_final = data_norm.T.astype(np.float32)

        # Augmentation: Channel Dropout (Train only)
        if self.augment and np.random.rand() < 0.5:
            # Drop 1 to 3 channels randomly
            num_drop = np.random.randint(1, 4)
            drop_indices = np.random.choice(
                data_final.shape[0], num_drop, replace=False
            )
            data_final[drop_indices, :] = 0.0

        return torch.tensor(data_final)

    def _load_spectrogram(self, row):
        """
        Loads spectrogram, extracts 4 regions, resizes, and injects coordinate channel.
        """
        spec_path = os.path.join(Config.INPUT_DIR, row["spectrogram_path"])

        try:
            spec_df = pd.read_parquet(spec_path)
        except Exception:
            return torch.zeros(
                (Config.SPEC_CHANNELS, Config.SPEC_SIZE[0], Config.SPEC_SIZE[1]),
                dtype=torch.float32,
            )

        # Determine Time Slice
        # Spectrogram rows are ~2 seconds (0.5 Hz)
        # 10 minutes = 600 seconds = 300 rows
        if self.mode == "test":
            start_row = 0
        else:
            offset_sec = row["spectrogram_label_offset_seconds"]
            start_row = int(offset_sec / 2.0)

        target_rows = 300
        end_row = start_row + target_rows

        # Handle bounds
        max_rows = len(spec_df)
        start_row = max(0, min(start_row, max_rows - target_rows))
        end_row = start_row + target_rows

        # Slice
        spec_slice = spec_df.iloc[start_row:end_row]

        # Pad if short
        if len(spec_slice) < target_rows:
            pad_rows = target_rows - len(spec_slice)
            pad_df = pd.DataFrame(0.0, index=range(pad_rows), columns=spec_df.columns)
            spec_slice = pd.concat([spec_slice, pad_df], axis=0)

        # Extract 4 Regions (LL, RL, LP, RP)
        # Columns are like "LL_0.59", "LL_0.78", etc.
        all_cols = spec_slice.columns
        regions = []

        for prefix in SPEC_REGION_PREFIXES:
            # Identify columns for this region
            r_cols = [c for c in all_cols if c.startswith(f"{prefix}_")]

            if not r_cols:
                # Fallback if columns missing
                region_data = np.zeros((target_rows, 100), dtype=np.float32)
            else:
                # Sort by frequency (value after underscore)
                try:
                    r_cols = sorted(r_cols, key=lambda x: float(x.split("_")[1]))
                except:
                    pass  # Use default order if parse fails

                region_data = spec_slice[r_cols].values

            # Log Transform and Normalize
            # log1p(x) is standard for power spectrograms
            region_data = np.log1p(np.nan_to_num(region_data, nan=0.0))
            regions.append(region_data)

        # Resize and Stack
        # Target: (512, 512)
        resized_regions = []
        for r in regions:
            # cv2.resize(src, dsize=(width, height))
            # r is (Time, Freq). We want output (512, 512).
            # We map Time -> Height, Freq -> Width.
            r_resized = cv2.resize(r, Config.SPEC_SIZE, interpolation=cv2.INTER_LINEAR)
            resized_regions.append(r_resized)

        # Stack 4 regions: (512, 512, 4)
        spec_img = np.stack(resized_regions, axis=-1)

        # Normalize [0, 1] or Standardize
        # Using standardization
        mean = spec_img.mean()
        std = spec_img.std()
        spec_img = (spec_img - mean) / (std + 1e-6)

        # Transpose to (Channels, Height, Width) -> (4, 512, 512)
        spec_img = spec_img.transpose(2, 0, 1)

        # Coordinate Injection (5th Channel)
        # Gradient from -1 to 1 along Height (Time)
        H, W = Config.SPEC_SIZE
        coord_vec = np.linspace(-1, 1, H, dtype=np.float32)
        # Broadcast to (H, W)
        coord_map = np.tile(coord_vec[:, None], (1, W))  # (512, 512)
        coord_map = coord_map[None, :, :]  # (1, 512, 512)

        # Concatenate -> (5, 512, 512)
        final_spec = np.concatenate([spec_img, coord_map], axis=0)

        # Augmentation: SpecAugment (Train only)
        if self.augment:
            # Apply masking to first 4 channels
            C_total, H_img, W_img = final_spec.shape

            # Time Masking
            if np.random.rand() < 0.5:
                t_width = np.random.randint(0, H_img // 8)
                t_start = np.random.randint(0, H_img - t_width)
                final_spec[:4, t_start : t_start + t_width, :] = 0.0

            # Frequency Masking
            if np.random.rand() < 0.5:
                f_width = np.random.randint(0, W_img // 8)
                f_start = np.random.randint(0, W_img - f_width)
                final_spec[:4, :, f_start : f_start + f_width] = 0.0

        return torch.tensor(final_spec.astype(np.float32))


def get_dataloaders(
    train_batch_size=Config.BATCH_SIZE,
    val_batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    debug_limit=None,
):
    """
    Creates DataLoaders for train, validation, and test sets.
    """
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Optional: Limit data for debugging
    if debug_limit:
        train_df = train_df.iloc[:debug_limit]
        val_df = val_df.iloc[:debug_limit]
        # Keep test full usually, but can limit if needed
        # test_df = test_df.iloc[:debug_limit]

    # Initialize Datasets
    train_ds = MultiModalDataset(train_df, mode="train", augment=True)
    val_ds = MultiModalDataset(val_df, mode="val", augment=False)
    test_ds = MultiModalDataset(test_df, mode="test", augment=False)

    # Initialize DataLoaders
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
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
