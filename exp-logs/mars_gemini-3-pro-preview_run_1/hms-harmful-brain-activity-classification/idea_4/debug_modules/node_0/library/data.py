import os
import torch
import numpy as np
import pandas as pd
import cv2
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# Standard EEG channel names corresponding to the 10-20 system + EKG
# Used to ensure consistent channel ordering across different files
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


class EEGSpecDataset(Dataset):
    """
    Dual-stream dataset for loading paired EEG signals and Spectrogram images.
    """

    def __init__(self, metadata, mode="train", augment=False):
        self.metadata = metadata
        self.mode = mode
        self.augment = augment

        # Pre-compute full paths for efficiency
        self.metadata["eeg_full_path"] = self.metadata["eeg_path"].apply(
            lambda x: os.path.join(Config.INPUT_DIR, x)
        )
        self.metadata["spec_full_path"] = self.metadata["spectrogram_path"].apply(
            lambda x: os.path.join(Config.INPUT_DIR, x)
        )

        # Spectrogram Augmentations (SpecAugment approximation)
        if self.augment:
            self.spec_transform = A.Compose(
                [
                    A.CoarseDropout(
                        max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.5
                    ),
                ]
            )
        else:
            self.spec_transform = None

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # 1. Load and Process EEG Signal
        eeg_tensor = self._load_eeg(row)

        # 2. Load and Process Spectrogram Image
        spec_tensor = self._load_spectrogram(row)

        # 3. Return Data
        if self.mode != "test":
            # Return (Inputs, Targets)
            targets = row[Config.TARGET_COLS].values.astype(np.float32)
            return (eeg_tensor, spec_tensor), torch.tensor(targets)
        else:
            # Return Inputs only
            return (eeg_tensor, spec_tensor)

    def _load_eeg(self, row):
        path = row["eeg_full_path"]
        offset = row["eeg_label_offset_seconds"]

        try:
            # Read Parquet file
            # Note: Reading the whole file is often safer/easier than partial reads with parquet
            # unless the file is massive, but these are manageable.
            df = pd.read_parquet(path)

            # Calculate indices for the 50s window at 200Hz
            start_idx = int(offset * 200)
            end_idx = start_idx + int(Config.EEG_DURATION * 200)  # 10000 samples

            # Handle boundary conditions
            if start_idx < 0:
                start_idx = 0

            # Slice the dataframe
            segment = df.iloc[start_idx:end_idx]

            # Filter columns
            available_cols = segment.columns.tolist()
            # Exclude 'time' column if present
            data_cols = [c for c in available_cols if c != "time"]

            # Select channels
            selected_cols = []
            for col in EEG_CHANNELS:
                if col in data_cols:
                    selected_cols.append(col)

            # Fallback if specific channels are missing (take first 20)
            if len(selected_cols) < Config.EEG_CHANNELS:
                selected_cols = data_cols[: Config.EEG_CHANNELS]

            signal = segment[selected_cols].values

            # Pad with zeros if segment is shorter than expected
            expected_len = int(Config.EEG_DURATION * 200)
            if len(signal) < expected_len:
                pad_len = expected_len - len(signal)
                signal = np.pad(signal, ((0, pad_len), (0, 0)), mode="constant")

            # Handle NaNs
            signal = np.nan_to_num(signal, nan=0.0)

            # Clip outliers (artifact removal)
            signal = np.clip(signal, -1024, 1024)

            # Downsample to 100 Hz (Take every 2nd sample)
            signal = signal[::2, :]  # Shape: (5000, 20)

            # Transpose to (Channels, Time) -> (20, 5000)
            signal = signal.transpose(1, 0)

            # Instance Normalization (per channel)
            mean = np.mean(signal, axis=1, keepdims=True)
            std = np.std(signal, axis=1, keepdims=True)
            signal = (signal - mean) / (std + 1e-6)

            # Augmentation: Channel Dropout
            if self.augment and np.random.rand() < 0.5:
                # Randomly drop 1 to 3 channels
                num_drop = np.random.randint(1, 4)
                channels_to_drop = np.random.choice(
                    signal.shape[0], num_drop, replace=False
                )
                signal[channels_to_drop, :] = 0.0

            return torch.tensor(signal, dtype=torch.float32)

        except Exception as e:
            # Return zero tensor on failure
            return torch.zeros(
                (Config.EEG_CHANNELS, Config.EEG_SEQ_LEN), dtype=torch.float32
            )

    def _load_spectrogram(self, row):
        path = row["spec_full_path"]
        offset = row["spectrogram_label_offset_seconds"]

        try:
            df = pd.read_parquet(path)

            # Extract 10-minute window (600 seconds)
            if "time" in df.columns:
                mask = (df["time"] >= offset) & (df["time"] < offset + 600)
                segment = df.loc[mask]
                data = segment.drop(columns=["time"]).values
            else:
                # Fallback: use all data
                data = df.values

            # Handle NaNs
            data = np.nan_to_num(data, nan=0.0)

            # Log transformation
            data = np.log1p(data)

            # Resize to target resolution (Width, Height)
            # data shape is (Time, Freq), cv2.resize expects (W, H)
            data = cv2.resize(
                data,
                (Config.SPEC_SIZE[1], Config.SPEC_SIZE[0]),
                interpolation=cv2.INTER_LINEAR,
            )

            # Min-Max Normalization
            d_min = data.min()
            d_max = data.max()
            if d_max > d_min:
                data = (data - d_min) / (d_max - d_min)
            else:
                data = np.zeros_like(data)

            # Convert to 3 channels (RGB) for EfficientNet
            data = np.stack([data, data, data], axis=-1)

            # Augmentation
            if self.augment and self.spec_transform:
                res = self.spec_transform(image=data)
                data = res["image"]

            # Transpose to (Channels, Height, Width) -> (3, 512, 512)
            data = data.transpose(2, 0, 1)

            return torch.tensor(data, dtype=torch.float32)

        except Exception as e:
            # Return zero tensor on failure
            return torch.zeros(
                (3, Config.SPEC_SIZE[0], Config.SPEC_SIZE[1]), dtype=torch.float32
            )


def get_dataloaders(
    train_batch_size=Config.BATCH_SIZE,
    val_batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    debug=Config.DEBUG,
    debug_subset_size=Config.DEBUG_SUBSET_SIZE,
):
    """
    Creates and returns DataLoaders for train, validation, and test sets.
    """
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Debug Mode: Sample subset
    if debug:
        train_df = train_df.sample(
            n=min(len(train_df), debug_subset_size), random_state=Config.SEED
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(len(val_df), debug_subset_size), random_state=Config.SEED
        ).reset_index(drop=True)
        test_df = test_df.sample(
            n=min(len(test_df), debug_subset_size), random_state=Config.SEED
        ).reset_index(drop=True)

    # Instantiate Datasets
    train_dataset = EEGSpecDataset(train_df, mode="train", augment=True)
    val_dataset = EEGSpecDataset(val_df, mode="val", augment=False)
    test_dataset = EEGSpecDataset(test_df, mode="test", augment=False)

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
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
