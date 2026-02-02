import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from library.config import AudioConfig, AugmentConfig, PathConfig, TrainConfig
from library.utils import set_seed

# Ensure reproducibility
set_seed(42)


class WhaleDataset(Dataset):
    """
    Custom Dataset for Right Whale Detection.
    Wraps cached spectrogram data and applies on-the-fly augmentation and normalization.
    """

    def __init__(
        self, data, targets=None, transform=None, is_test=False, clip_names=None
    ):
        """
        Args:
            data (np.ndarray): Array of spectrograms (N, Freq, Time).
            targets (np.ndarray, optional): Array of labels (N,).
            transform (nn.Module, optional): Augmentation transforms.
            is_test (bool): Whether this is the test set.
            clip_names (np.ndarray, optional): Filenames for test set identification.
        """
        self.data = data
        self.targets = targets
        self.transform = transform
        self.is_test = is_test
        self.clip_names = clip_names

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve spectrogram: (Freq, Time)
        spec = self.data[idx]

        # Convert to Tensor
        spec = torch.from_numpy(spec)

        # Add channel dimension: (1, Freq, Time)
        spec = spec.unsqueeze(0)

        # Apply Augmentations (only for training)
        if self.transform:
            spec = self.transform(spec)

        # Instance Normalization (Standardization)
        # Robust scaling for varying signal levels
        mean = spec.mean()
        std = spec.std()
        if std > 1e-6:
            spec = (spec - mean) / std
        else:
            spec = spec - mean

        if self.is_test:
            return spec, self.clip_names[idx]
        else:
            # Return label as float for BCEWithLogitsLoss
            return spec, torch.tensor(self.targets[idx], dtype=torch.float32)


def get_augment_transform():
    """
    Returns the SpecAugment pipeline based on AugmentConfig.
    """
    return torch.nn.Sequential(
        T.TimeMasking(time_mask_param=AugmentConfig.TIME_MASK_PARAM),
        T.FrequencyMasking(freq_mask_param=AugmentConfig.FREQ_MASK_PARAM),
    )


def process_audio(filepath, mel_transform, db_transform, resampler=None):
    """
    Loads and processes a single audio file into a Log-Mel Spectrogram.
    """
    try:
        # Load audio
        waveform, sr = torchaudio.load(filepath)
    except Exception:
        # Fallback for corrupt files: return silent spectrogram
        target_len = int(AudioConfig.SAMPLE_RATE * AudioConfig.DURATION)
        # Calculate time steps based on hop length
        n_steps = int(target_len / AudioConfig.HOP_LENGTH) + 1
        return torch.zeros(AudioConfig.N_MELS, n_steps)

    # Resample if necessary
    if sr != AudioConfig.SAMPLE_RATE:
        if resampler is None:
            resampler = T.Resample(sr, AudioConfig.SAMPLE_RATE)
        waveform = resampler(waveform)

    # Convert to Mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Pad or Crop to fixed duration (2.0s)
    target_samples = int(AudioConfig.SAMPLE_RATE * AudioConfig.DURATION)
    current_samples = waveform.shape[1]

    if current_samples < target_samples:
        padding = target_samples - current_samples
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    elif current_samples > target_samples:
        waveform = waveform[:, :target_samples]

    # Compute Log-Mel Spectrogram
    spec = mel_transform(waveform)
    spec = db_transform(spec)

    # Remove channel dim for storage efficiency: (1, F, T) -> (F, T)
    return spec.squeeze(0)


def load_and_cache_data(csv_path, cache_name, load_cached_data=True):
    """
    Loads metadata, processes audio files to spectrograms, and caches them as .npy files.
    """
    cache_dir = PathConfig.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    data_path = os.path.join(cache_dir, f"{cache_name}_data.npy")
    targets_path = os.path.join(cache_dir, f"{cache_name}_targets.npy")
    ids_path = os.path.join(cache_dir, f"{cache_name}_ids.npy")

    # Attempt to load from cache
    if load_cached_data and os.path.exists(data_path) and os.path.exists(ids_path):
        print(f"Loading cached {cache_name} data from {cache_dir}...")
        data = np.load(data_path)
        ids = np.load(ids_path)
        targets = None
        if os.path.exists(targets_path):
            targets = np.load(targets_path)
        return data, targets, ids

    print(f"Processing {cache_name} data from scratch...")

    # Load Metadata
    df = pd.read_csv(csv_path)

    # Initialize Transforms
    mel_transform = T.MelSpectrogram(
        sample_rate=AudioConfig.SAMPLE_RATE,
        n_fft=AudioConfig.N_FFT,
        hop_length=AudioConfig.HOP_LENGTH,
        n_mels=AudioConfig.N_MELS,
        f_min=AudioConfig.F_MIN,
        f_max=AudioConfig.F_MAX,
        normalized=True,
    )

    db_transform = T.AmplitudeToDB(top_db=AudioConfig.TOP_DB)

    data_list = []
    target_list = []
    id_list = []

    for _, row in df.iterrows():
        full_path = os.path.join(PathConfig.INPUT_ROOT, row["filepath"])

        # Process
        spec = process_audio(full_path, mel_transform, db_transform)

        data_list.append(spec.numpy())
        id_list.append(row["clip"])

        if "label" in row:
            target_list.append(row["label"])

    # Stack and Save
    data_np = np.stack(data_list).astype(np.float32)
    ids_np = np.array(id_list)

    print(f"Saving {cache_name} data to {cache_dir}...")
    np.save(data_path, data_np)
    np.save(ids_path, ids_np)

    targets_np = None
    if target_list:
        targets_np = np.array(target_list).astype(np.int64)
        np.save(targets_path, targets_np)

    return data_np, targets_np, ids_np


def get_dataloaders(load_cached_data=True):
    """
    Constructs and returns DataLoaders for Train, Validation, and Test sets.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed .npy files.
                                 If False, re-processes raw audio.
    """
    # 1. Load Data
    train_data, train_targets, _ = load_and_cache_data(
        PathConfig.TRAIN_CSV, "train", load_cached_data
    )

    val_data, val_targets, _ = load_and_cache_data(
        PathConfig.VAL_CSV, "val", load_cached_data
    )

    test_data, _, test_ids = load_and_cache_data(
        PathConfig.TEST_CSV, "test", load_cached_data
    )

    # 2. Create Datasets
    # Apply SpecAugment only to Training set
    train_dataset = WhaleDataset(
        train_data, train_targets, transform=get_augment_transform(), is_test=False
    )

    val_dataset = WhaleDataset(val_data, val_targets, transform=None, is_test=False)

    test_dataset = WhaleDataset(
        test_data, targets=None, transform=None, is_test=True, clip_names=test_ids
    )

    # 3. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=TrainConfig.BATCH_SIZE,
        shuffle=True,
        num_workers=TrainConfig.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=TrainConfig.BATCH_SIZE,
        shuffle=False,
        num_workers=TrainConfig.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=TrainConfig.BATCH_SIZE,
        shuffle=False,
        num_workers=TrainConfig.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
