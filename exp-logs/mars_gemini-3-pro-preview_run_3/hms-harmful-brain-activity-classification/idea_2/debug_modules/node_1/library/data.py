import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from library import config
from library import utils

# Ensure reproducibility
utils.seed_everything(config.SEED)


class EEGDataset(Dataset):
    """
    Dataset class for loading EEG data, converting to Mel-Spectrograms,
    and preparing for 2D CNN input.
    """

    def __init__(self, df, mode="train", augment=False):
        self.df = df
        self.mode = mode
        self.augment = augment

        # Audio transform definition
        # We define it here but it runs on CPU within the worker processes
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.SAMPLING_RATE,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS,
            f_min=config.FMIN,
            f_max=config.FMAX,
            center=True,
        )

        # Augmentations
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=20)
        self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=40)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Resolve File Path
        # Metadata contains relative paths (e.g., 'train_eegs/123.parquet')
        file_path = os.path.join(config.INPUT_DIR, row["eeg_path"])

        # 2. Load EEG Signal
        try:
            # Read specific channels from parquet
            eeg_df = pd.read_parquet(file_path, columns=config.EEG_CHANNELS)
            raw_data = eeg_df.values  # Shape: (Total_Time, 19)
        except Exception as e:
            # Fallback for corrupted files
            raw_data = np.zeros(
                (config.TOTAL_SAMPLES, config.N_CHANNELS), dtype=np.float32
            )

        # 3. Extract 50s Window
        # Determine start index based on offset
        if self.mode == "test":
            # Test files are exactly 50s
            start_idx = 0
        else:
            # Train files are longer, use offset
            offset_sec = row.get("eeg_label_offset_seconds", 0)
            start_idx = int(offset_sec * config.SAMPLING_RATE)

        end_idx = start_idx + config.TOTAL_SAMPLES

        # Handle bounds (padding or cropping)
        total_available = raw_data.shape[0]

        if start_idx < 0:
            start_idx = 0
            end_idx = config.TOTAL_SAMPLES

        if end_idx <= total_available:
            window = raw_data[start_idx:end_idx, :]
        else:
            # Pad if window goes out of bounds
            window = raw_data[start_idx:, :]
            pad_len = config.TOTAL_SAMPLES - window.shape[0]
            if pad_len > 0:
                padding = np.zeros((pad_len, config.N_CHANNELS), dtype=window.dtype)
                window = np.concatenate([window, padding], axis=0)
            else:
                # Fallback if start_idx is completely out of bounds
                window = np.zeros(
                    (config.TOTAL_SAMPLES, config.N_CHANNELS), dtype=np.float32
                )

        # 4. Handle NaNs (Channel Mean Imputation)
        # Iterate channels to fill NaNs with that channel's mean
        for c in range(config.N_CHANNELS):
            col = window[:, c]
            if np.isnan(col).any():
                mean_val = np.nanmean(col)
                if np.isnan(mean_val):
                    mean_val = 0.0
                col[np.isnan(col)] = mean_val
                window[:, c] = col

        # 5. Signal to Spectrogram
        # Convert to tensor and transpose to (Channels, Time)
        tensor_eeg = torch.tensor(window, dtype=torch.float32).t()  # (19, 10000)

        # Compute Mel Spectrogram -> (19, 128, ~157)
        spec = self.mel_spectrogram(tensor_eeg)

        # Log Transform: log(S + eps)
        spec = torch.log(spec + 1e-6)

        # 6. Stack Channels Vertically
        # Reshape (Channels, Mels, Time) -> (Channels * Mels, Time)
        # This creates a tall image where y-axis is frequency across all leads
        c, m, t = spec.shape
        spec_image = spec.reshape(c * m, t)  # (2432, 157)

        # Add batch/channel dimension for interpolation: (1, 1, H, W)
        spec_image = spec_image.unsqueeze(0).unsqueeze(0)

        # 7. Resize to Target Image Size
        spec_image = F.interpolate(
            spec_image, size=config.IMG_SIZE, mode="bilinear", align_corners=False
        ).squeeze(
            0
        )  # Result: (1, 512, 512)

        # 8. Augmentation (Training Only)
        if self.augment:
            spec_image = self.freq_masking(spec_image)
            spec_image = self.time_masking(spec_image)

        # 9. Normalization (Instance Z-Score)
        # Standardize the image to mean 0, std 1
        mean = spec_image.mean()
        std = spec_image.std()
        if std > 1e-6:
            spec_image = (spec_image - mean) / std
        else:
            spec_image = spec_image - mean

        # 10. Replicate Channels (1 -> 3)
        # EfficientNet expects 3 input channels
        spec_image = spec_image.repeat(3, 1, 1)

        # 11. Return Data and Target
        if self.mode != "test":
            target = row[config.TARGET_COLS].values.astype(np.float32)
            return spec_image, torch.tensor(target)
        else:
            # For test set, return dummy target
            return spec_image, torch.tensor(0.0)


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
