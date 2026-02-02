import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything


class EEGDataset(Dataset):
    """
    PyTorch Dataset for EEG Harmful Brain Activity Detection.

    Features:
    - Loads pre-processed raw EEG signals (19 channels, 50 seconds).
    - Converts raw signals to Log-Mel Spectrograms on-the-fly.
    - Resizes spectrograms to fixed dimensions (512x512).
    - Applies SpecAugment (Time/Frequency Masking) during training.
    """

    def __init__(self, data, targets=None, mode="train"):
        """
        Args:
            data (np.ndarray): Array of shape (N, 19, 10000) containing raw EEG signals.
            targets (np.ndarray or None): Array of shape (N, 6) containing target probabilities.
            mode (str): 'train', 'val', or 'test'. Controls augmentation.
        """
        self.data = data
        self.targets = targets
        self.mode = mode

        # ==========================================
        # Transforms
        # ==========================================
        # 1. MelSpectrogram: Raw Audio -> Time-Frequency Representation
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.SAMPLING_RATE,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            f_min=Config.FMIN,
            f_max=Config.FMAX,
            power=2.0,
            normalized=True,
        )

        # 2. AmplitudeToDB: Convert power to decibels (Log scale)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80.0)

        # 3. SpecAugment (Training Only)
        # Parameters chosen to be moderate regularizers
        self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=30)
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=20)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # ------------------------------------------------------------------
        # 1. Load Raw Signal
        # ------------------------------------------------------------------
        # Shape: (19, 10000)
        eeg_signal = self.data[idx]

        # Convert to tensor
        waveform = torch.tensor(eeg_signal, dtype=torch.float32)

        # Safety: Handle any remaining NaNs (though preprocessing should have handled them)
        if torch.isnan(waveform).any():
            waveform = torch.nan_to_num(waveform, nan=0.0)

        # ------------------------------------------------------------------
        # 2. Generate Spectrogram
        # ------------------------------------------------------------------
        # Input: (19, 10000) -> Output: (19, n_mels, time_steps)
        # With 10000 samples and hop=20, time_steps ~= 501
        spec = self.mel_spec(waveform)
        spec = self.amplitude_to_db(spec)

        # ------------------------------------------------------------------
        # 3. Resize / Interpolate
        # ------------------------------------------------------------------
        # We need to resize to Config.IMG_SIZE (e.g., 512x512).
        # F.interpolate expects (Batch, Channels, H, W).
        # We treat the 19 EEG channels as the 'Batch' dimension for interpolation purposes
        # to resize each channel's spectrogram independently.
        # Shape: (19, 128, 501) -> (1, 19, 128, 501)
        spec = spec.unsqueeze(0)

        # Resize to (512, 512)
        spec = F.interpolate(
            spec, size=Config.IMG_SIZE, mode="bilinear", align_corners=False
        )

        # Remove dummy batch dim -> (19, 512, 512)
        spec = spec.squeeze(0)

        # ------------------------------------------------------------------
        # 4. Normalization
        # ------------------------------------------------------------------
        # Standardize each channel independently to mean=0, std=1
        mean = spec.mean(dim=(1, 2), keepdim=True)
        std = spec.std(dim=(1, 2), keepdim=True)
        spec = (spec - mean) / (std + 1e-6)

        # ------------------------------------------------------------------
        # 5. Augmentation (Train Only)
        # ------------------------------------------------------------------
        if self.mode == "train":
            # Apply SpecAugment
            spec = self.freq_masking(spec)
            spec = self.time_masking(spec)

        # ------------------------------------------------------------------
        # 6. Return
        # ------------------------------------------------------------------
        if self.targets is not None:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return spec, target
        else:
            return spec


def process_raw_eeg(df, mode="train"):
    """
    Reads individual Parquet files, extracts the specific 50-second window,
    and stacks them into a single NumPy array.

    Args:
        df (pd.DataFrame): Metadata dataframe.
        mode (str): 'train', 'val', or 'test'.

    Returns:
        np.ndarray: Array of shape (N_samples, 19, 10000)
    """
    n_samples = len(df)
    n_channels = Config.N_CHANNELS
    n_timepoints = Config.N_SAMPLES

    print(f"Processing {n_samples} EEG files for {mode} set...")

    # Pre-allocate memory
    data_array = np.zeros((n_samples, n_channels, n_timepoints), dtype=np.float32)

    # Determine base directory
    # Metadata paths are relative to input/ (e.g., "train_eegs/123.parquet")
    base_dir = Config.INPUT_DIR

    # Iterate through metadata
    for i, row in df.iterrows():
        file_path = os.path.join(base_dir, row["eeg_path"])

        try:
            # Read Parquet
            # Only read the necessary 19 columns to save I/O
            eeg_df = pd.read_parquet(file_path, columns=Config.EEG_CHANNELS)

            # Determine Extraction Window
            if mode == "test":
                # Test files are exactly 50s
                start_idx = 0
            else:
                # Train/Val files may be longer; use offset
                offset = row["eeg_label_offset_seconds"]
                start_idx = int(offset * Config.SAMPLING_RATE)

            end_idx = start_idx + n_timepoints

            # Extract Data
            full_len = len(eeg_df)

            if start_idx >= full_len:
                # Fallback for invalid offset (should not happen with clean metadata)
                segment = np.zeros((n_timepoints, n_channels))
            elif end_idx > full_len:
                # Pad if segment goes out of bounds
                temp = eeg_df.iloc[start_idx:].values
                pad_len = n_timepoints - len(temp)
                segment = np.pad(temp, ((0, pad_len), (0, 0)), mode="constant")
            else:
                # Normal extraction
                segment = eeg_df.iloc[start_idx:end_idx].values

            # Transpose to (Channels, Time) -> (19, 10000)
            segment = segment.T

            # Handle NaNs (replace with 0)
            segment = np.nan_to_num(segment, nan=0.0)

            # Store
            data_array[i] = segment.astype(np.float32)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            # Leave as zeros

        # Log progress
        if (i + 1) % 5000 == 0:
            print(f"  Processed {i + 1}/{n_samples} files")

    return data_array


def load_data(mode="train", load_cached_data=True, debug_subset=None):
    """
    Loads metadata and raw EEG data. Implements caching logic.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load .npy files from working dir.
        debug_subset (int or None): If set, limits data to N samples.

    Returns:
        tuple: (DataFrame, data_array, targets_array)
    """
    # 1. Load Metadata
    if mode == "train":
        meta_path = Config.TRAIN_CSV
    elif mode == "val":
        meta_path = Config.VAL_CSV
    elif mode == "test":
        meta_path = Config.TEST_CSV
    else:
        raise ValueError(f"Unknown mode: {mode}")

    df = pd.read_csv(meta_path)

    # Apply Debug Subset
    if debug_subset is not None:
        df = df.head(debug_subset).copy()
        print(f"DEBUG: Subsetting {mode} data to {len(df)} rows.")

    # Reset index to ensure alignment with numpy arrays
    df.reset_index(drop=True, inplace=True)

    # 2. Define Cache Paths
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    suffix = "_debug" if debug_subset is not None else ""
    data_cache_path = os.path.join(Config.WORKING_DIR, f"{mode}_data{suffix}.npy")
    targets_cache_path = os.path.join(Config.WORKING_DIR, f"{mode}_targets{suffix}.npy")

    # 3. Handle Targets
    targets = None
    if mode != "test":
        # Extract targets from dataframe
        targets = df[Config.TARGET_COLS].values.astype(np.float32)

        # Cache targets (optional, but good for consistency check)
        if not os.path.exists(targets_cache_path) or not load_cached_data:
            np.save(targets_cache_path, targets)

    # 4. Handle Data (EEG Signals)
    data = None

    # Try loading cache
    if load_cached_data and os.path.exists(data_cache_path):
        print(f"Loading cached {mode} data from {data_cache_path}...")
        try:
            data = np.load(data_cache_path)
            # Verify shape matches metadata
            if len(data) != len(df):
                print(
                    f"Cache size mismatch ({len(data)} vs {len(df)}). Reprocessing..."
                )
                data = None
            else:
                print("Cache loaded successfully.")
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")
            data = None

    # Process from scratch if needed
    if data is None:
        data = process_raw_eeg(df, mode=mode)
        print(f"Saving {mode} data to {data_cache_path}...")
        np.save(data_cache_path, data)

    return df, data, targets


def get_dataloader(
    mode="train", batch_size=Config.BATCH_SIZE, load_cached_data=True, debug_subset=None
):
    """
    Creates a PyTorch DataLoader for the specified mode.

    Args:
        mode (str): 'train', 'val', or 'test'.
        batch_size (int): Batch size.
        load_cached_data (bool): Whether to use cached .npy files.
        debug_subset (int or None): Limit dataset size for debugging.

    Returns:
        DataLoader: Configured PyTorch DataLoader.
    """
    # Load Data
    _, data, targets = load_data(mode, load_cached_data, debug_subset)

    # Create Dataset
    dataset = EEGDataset(data, targets, mode=mode)

    # Configure Loader
    shuffle = mode == "train"
    drop_last = mode == "train"

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=(Config.NUM_WORKERS > 0),
    )

    return loader
