import os
import numpy as np
import pandas as pd
import torch
import cv2
import librosa
from torch.utils.data import Dataset, DataLoader
from joblib import Parallel, delayed
from library.config import Config
from library.utils import seed_everything

# ==========================================
# Data Processing Functions
# ==========================================


def process_eeg_signal(eeg_signal, output_size=(128, 256)):
    """
    Converts raw EEG signal (Time, Channels) into a Band-Adaptive Multi-Resolution Tensor.
    Output Shape: (57, 128, 256) -> (Channels*Bands, Freq, Time)
    """
    # eeg_signal: (10000, 19)
    # Handle NaNs
    eeg_signal = np.nan_to_num(eeg_signal, nan=0.0)

    n_channels = eeg_signal.shape[1]
    generated_specs = []

    # Iterate over each channel
    for ch in range(n_channels):
        channel_data = eeg_signal[:, ch]  # (10000,)

        channel_specs = []

        # Apply 3 Band-Adaptive STFTs
        for band_conf in Config.STFT_BANDS:
            # Parameters
            sr = Config.EEG_SR
            n_fft = int(band_conf["window_sec"] * sr)
            hop_length = int(band_conf["hop_sec"] * sr)
            fmin = band_conf["fmin"]
            fmax = band_conf["fmax"]
            n_mels = band_conf["n_mels"]

            # Compute Mel Spectrogram
            # Power=2.0 for power spectrogram
            S = librosa.feature.melspectrogram(
                y=channel_data,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                fmin=fmin,
                fmax=fmax,
                power=2.0,
            )

            # Log transform (dB)
            S_db = librosa.power_to_db(S, ref=np.max)

            # Normalize to 0-1 range roughly or standard scaler?
            # Standard approach: (S - min) / (max - min) per image is risky if signal is flat.
            # We use global normalization in the dataset usually, but here we standardize per sample
            # to be robust to amplitude differences.
            mean = S_db.mean()
            std = S_db.std()
            if std > 1e-6:
                S_db = (S_db - mean) / std
            else:
                S_db = S_db - mean

            # Resize to fixed spatial dimensions for stacking
            # S_db shape is (n_mels, time_steps).
            # We resize to (128, 256) as defined in config, but wait.
            # The config says IMG_SIZE_A = (128, 256).
            # We have 3 bands per channel. If we stack them in depth (channel dim),
            # each band must be the same spatial size.
            # We resize each band's output to (128, 256).

            # cv2.resize expects (Width, Height)
            S_resized = cv2.resize(
                S_db, (output_size[1], output_size[0]), interpolation=cv2.INTER_LINEAR
            )
            channel_specs.append(S_resized)

        # Stack the 3 bands for this electrode
        # Each is (128, 256).
        generated_specs.extend(channel_specs)

    # Convert to tensor shape (C, H, W)
    # We have 19 channels * 3 bands = 57 items in generated_specs
    output_tensor = np.stack(generated_specs, axis=0)  # (57, 128, 256)

    return output_tensor.astype(np.float32)


def process_spectrogram(spec_data, output_size=(256, 256)):
    """
    Processes the 10m spectrogram data.
    Input: (Time, Freq_Cols) raw dataframe values
    Output: (4, 256, 256) Tensor
    """
    # spec_data is (Time, 401) where columns are LL_x, RL_x, LP_x, RP_x mixed + time
    # We assume the input passed here is already the slice of values without the 'time' column if processed correctly,
    # or we handle the raw array.
    # Based on EDA, there are 400 freq columns + 1 time column usually.
    # We will assume the loader extracts the 4 regions properly.

    # Actually, parsing the columns for every sample is slow.
    # We will assume the loader provides a (Time, 400) array where:
    # 0-99: LL, 100-199: RL, 200-299: LP, 300-399: RP (Standard for this dataset)
    # We handle NaNs first
    spec_data = np.nan_to_num(spec_data, nan=0.0)

    # Log transform
    # Clip to avoid log(0)
    spec_data = np.clip(spec_data, np.exp(-4), None)
    spec_data = np.log(spec_data)

    # Split into 4 regions
    n_freqs = spec_data.shape[1] // 4
    regions = []
    for i in range(4):
        start = i * n_freqs
        end = (i + 1) * n_freqs
        region = spec_data[:, start:end]  # (Time, 100)

        # Transpose to (Freq, Time) -> (100, Time)
        region = region.T

        # Standardize
        mean = region.mean()
        std = region.std()
        if std > 1e-6:
            region = (region - mean) / std
        else:
            region = region - mean

        # Resize
        region_resized = cv2.resize(
            region, (output_size[1], output_size[0]), interpolation=cv2.INTER_LINEAR
        )
        regions.append(region_resized)

    output_tensor = np.stack(regions, axis=0)  # (4, 256, 256)
    return output_tensor.astype(np.float32)


# ==========================================
# Data Loading & Caching
# ==========================================


def _load_single_eeg(row, base_dir):
    """Helper to load and slice a single EEG file."""
    try:
        eeg_path = os.path.join(base_dir, row["eeg_path"])
        df = pd.read_parquet(eeg_path, columns=Config.EEG_CHANNELS)

        # Slice
        if "eeg_label_offset_seconds" in row:
            offset = int(row["eeg_label_offset_seconds"] * Config.EEG_SR)
            data = df.iloc[offset : offset + Config.EEG_SAMPLES].values
        else:
            # Test set: take full file (should be 50s)
            data = df.values

        # Pad if short (rare edge case)
        if len(data) < Config.EEG_SAMPLES:
            pad = np.zeros((Config.EEG_SAMPLES - len(data), data.shape[1]))
            data = np.concatenate([data, pad], axis=0)
        elif len(data) > Config.EEG_SAMPLES:
            data = data[: Config.EEG_SAMPLES]

        return data.astype(np.float32)
    except Exception as e:
        # Return zeros in case of failure to avoid crashing
        return np.zeros(
            (Config.EEG_SAMPLES, len(Config.EEG_CHANNELS)), dtype=np.float32
        )


def _load_single_spec(row, base_dir):
    """Helper to load and slice a single Spectrogram file."""
    try:
        spec_path = os.path.join(base_dir, row["spec_path"])
        df = pd.read_parquet(spec_path)

        # Identify region columns (exclude 'time')
        # We assume columns are sorted or standard.
        # Standard Kaggle HMS: LL, RL, LP, RP blocks.
        # We filter out 'time'
        cols = [c for c in df.columns if "time" not in c]

        if "spectogram_label_offset_seconds" in row:
            offset = row["spectogram_label_offset_seconds"]
            # Assume 'time' column exists for slicing
            if "time" in df.columns:
                # Slice 600 seconds (10 mins)
                subset = df.loc[(df["time"] >= offset) & (df["time"] < offset + 600)]
                data = subset[cols].values
            else:
                # Fallback: Assume file is the segment
                data = df[cols].values
        else:
            # Test set
            data = df[cols].values

        # Resize/Pad time dimension to a fixed size for raw storage?
        # No, we store raw variable length and resize in process_spectrogram.
        # But for numpy stacking, we need fixed size.
        # Let's resize the TIME dimension to fixed 300 (approx mean) or 400 here?
        # Better: Resize to fixed size (e.g. 400 time steps) to allow stacking.
        # The process_spectrogram will resize to 256x256 anyway.

        # Current shape: (Time, 400)
        target_time_steps = 400
        if data.shape[0] != target_time_steps:
            # Resize (Width=400 freq, Height=TargetTime)
            # cv2.resize takes (Width, Height). Input is (Time, Freq)
            # We want (TargetTime, Freq)
            data = cv2.resize(
                data,
                (data.shape[1], target_time_steps),
                interpolation=cv2.INTER_NEAREST,
            )

        return data.astype(np.float32)
    except Exception as e:
        return np.zeros((400, 400), dtype=np.float32)  # Fallback


def load_data(mode="train", load_cached_data=True):
    """
    Loads data for the specified mode.
    Uses caching to store pre-sliced raw data as .npy files.
    """
    print(f"Loading data for mode: {mode}")

    # Define paths
    if mode == "train":
        csv_path = Config.TRAIN_CSV
        eeg_cache = Config.CACHE_FILES["train_eeg"]
        spec_cache = Config.CACHE_FILES["train_spec"]
        target_cache = os.path.join(Config.WORKING_DIR, "train_targets.npy")
        base_dir = Config.INPUT_DIR
    elif mode == "val":
        csv_path = Config.VAL_CSV
        eeg_cache = Config.CACHE_FILES["val_eeg"]
        spec_cache = Config.CACHE_FILES["val_spec"]
        target_cache = os.path.join(Config.WORKING_DIR, "val_targets.npy")
        base_dir = Config.INPUT_DIR
    else:  # test
        csv_path = Config.TEST_CSV
        eeg_cache = Config.CACHE_FILES["test_eeg"]
        spec_cache = Config.CACHE_FILES["test_spec"]
        target_cache = None
        base_dir = Config.INPUT_DIR

    # Check cache
    if load_cached_data and os.path.exists(eeg_cache) and os.path.exists(spec_cache):
        if mode != "test" and not os.path.exists(target_cache):
            pass  # Targets missing, reload
        else:
            print(f"Loading cached data from {Config.WORKING_DIR}...")
            eeg_data = np.load(eeg_cache, mmap_mode="r")
            spec_data = np.load(spec_cache, mmap_mode="r")
            if mode != "test":
                targets = np.load(target_cache)
                return eeg_data, spec_data, targets
            return eeg_data, spec_data, None

    # Load Metadata
    df = pd.read_csv(csv_path)
    if Config.DEBUG:
        df = df.head(100)
        print("DEBUG MODE: Loading only 100 samples.")

    # Parallel Load EEG
    print(f"Processing {len(df)} EEG files...")
    rows = df.to_dict("records")
    eeg_data = Parallel(n_jobs=Config.NUM_WORKERS, backend="loky")(
        delayed(_load_single_eeg)(row, base_dir) for row in rows
    )
    eeg_data = np.array(eeg_data, dtype=np.float32)

    # Parallel Load Spectrograms
    print(f"Processing {len(df)} Spectrogram files...")
    spec_data = Parallel(n_jobs=Config.NUM_WORKERS, backend="loky")(
        delayed(_load_single_spec)(row, base_dir) for row in rows
    )
    spec_data = np.array(spec_data, dtype=np.float32)

    # Save to Cache
    print("Saving to cache...")
    np.save(eeg_cache, eeg_data)
    np.save(spec_cache, spec_data)

    if mode != "test":
        targets = df[Config.TARGET_COLS].values.astype(np.float32)
        np.save(target_cache, targets)
        return eeg_data, spec_data, targets

    return eeg_data, spec_data, None


# ==========================================
# Dataset Class
# ==========================================


class EEGDataset(Dataset):
    def __init__(self, eeg_data, spec_data, targets=None, mode="train", transform=None):
        self.eeg_data = eeg_data
        self.spec_data = spec_data
        self.targets = targets
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        # 1. Load Raw Data
        # If mmap, this reads from disk
        raw_eeg = self.eeg_data[idx]  # (10000, 19)
        raw_spec = self.spec_data[idx]  # (400, 400)

        # 2. Process EEG (Band-Adaptive)
        # Returns (57, 128, 256)
        X_eeg = process_eeg_signal(raw_eeg, output_size=Config.IMG_SIZE_A)

        # 3. Process Spectrogram
        # Returns (4, 256, 256)
        X_spec = process_spectrogram(raw_spec, output_size=Config.IMG_SIZE_B)

        # 4. Augmentation (SpecAugment)
        if self.mode == "train":
            X_eeg = self.apply_spec_augment(X_eeg)
            X_spec = self.apply_spec_augment(X_spec)

        # 5. Convert to Tensor
        X_eeg = torch.tensor(X_eeg, dtype=torch.float32)
        X_spec = torch.tensor(X_spec, dtype=torch.float32)

        if self.targets is not None:
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            return (X_eeg, X_spec), y
        else:
            return (X_eeg, X_spec)

    def apply_spec_augment(self, x):
        """
        Applies simple time and frequency masking to numpy array (C, F, T).
        """
        # x shape: (C, Freq, Time)
        C, F, T = x.shape

        # Frequency Masking
        if Config.SPECAUG_MASK_FREQ > 0:
            f_mask_param = Config.SPECAUG_MASK_FREQ
            f0 = np.random.randint(0, F - f_mask_param)
            x[:, f0 : f0 + f_mask_param, :] = 0

        # Time Masking
        if Config.SPECAUG_MASK_TIME > 0:
            t_mask_param = Config.SPECAUG_MASK_TIME
            t0 = np.random.randint(0, T - t_mask_param)
            x[:, :, t0 : t0 + t_mask_param] = 0

        return x


# ==========================================
# Utilities
# ==========================================


def mixup_data(x_eeg, x_spec, y, alpha=0.4, device="cuda"):
    """
    Applies MixUp to the batch.
    Returns: mixed_x_eeg, mixed_x_spec, y_a, y_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x_eeg.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x_eeg = lam * x_eeg + (1 - lam) * x_eeg[index, :]
    mixed_x_spec = lam * x_spec + (1 - lam) * x_spec[index, :]

    y_a, y_b = y, y[index]
    return mixed_x_eeg, mixed_x_spec, y_a, y_b, lam


def get_transforms(mode="train"):
    # Augmentation handled inside Dataset for this architecture
    return None
