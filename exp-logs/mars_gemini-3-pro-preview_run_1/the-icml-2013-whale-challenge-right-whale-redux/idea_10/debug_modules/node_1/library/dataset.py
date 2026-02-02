import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchaudio import transforms as T

from library.config import Config
from library.utils import set_seed

# Ensure deterministic behavior
set_seed(Config.SEED)


def pad_or_crop(waveform, target_length):
    """
    Pad waveform with zeros or crop to target length.
    waveform: (channels, time)
    """
    c, t = waveform.shape
    if t < target_length:
        padding = target_length - t
        # Pad at the end
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    elif t > target_length:
        waveform = waveform[:, :target_length]
    return waveform


def compute_spectrogram(filepath):
    """
    Load audio and compute Log-Mel Spectrogram.
    Returns numpy array of shape (1, n_mels, time).
    """
    full_path = os.path.join(Config.INPUT_ROOT, filepath)

    # Load audio
    try:
        waveform, sr = torchaudio.load(full_path)
    except Exception:
        # Fallback for corrupted files (create silent waveform)
        target_len = int(Config.SAMPLE_RATE * Config.DURATION)
        waveform = torch.zeros(1, target_len)
        sr = Config.SAMPLE_RATE

    # Resample if necessary
    if sr != Config.SAMPLE_RATE:
        resampler = T.Resample(sr, Config.SAMPLE_RATE)
        waveform = resampler(waveform)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Pad or Crop to fixed duration
    target_samples = int(Config.SAMPLE_RATE * Config.DURATION)
    waveform = pad_or_crop(waveform, target_samples)

    # Compute Mel Spectrogram
    mel_transform = T.MelSpectrogram(
        sample_rate=Config.SAMPLE_RATE,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH,
        n_mels=Config.N_MELS,
        f_min=Config.F_MIN,
        f_max=Config.F_MAX,
    )

    mel_spec = mel_transform(waveform)

    # Convert to Log Scale (dB)
    db_transform = T.AmplitudeToDB(top_db=80)
    log_mel_spec = db_transform(mel_spec)

    return log_mel_spec.numpy()


def process_and_cache_data(csv_path, cache_prefix, load_cached_data=True):
    """
    Load metadata, compute spectrograms (or load from cache), and return arrays.
    """
    # Adjust cache prefix for debug mode to avoid collisions
    if Config.DEBUG:
        cache_prefix = f"{cache_prefix}_debug"

    # Define cache paths
    data_path = os.path.join(Config.WORKING_DIR, f"{cache_prefix}_data.npy")
    labels_path = os.path.join(Config.WORKING_DIR, f"{cache_prefix}_labels.npy")
    ids_path = os.path.join(Config.WORKING_DIR, f"{cache_prefix}_ids.npy")

    # Check if cache exists
    cache_exists = os.path.exists(data_path) and (
        os.path.exists(labels_path) or os.path.exists(ids_path)
    )

    if load_cached_data and cache_exists:
        data = np.load(data_path)

        if os.path.exists(labels_path):
            targets = np.load(labels_path)
            return data, targets, None
        else:
            ids = np.load(ids_path)
            return data, None, ids
    else:
        # Process from scratch
        df = pd.read_csv(csv_path)

        # Limit for debugging if configured
        if Config.DEBUG:
            df = df.head(Config.DEBUG_SUBSET_SIZE)

        specs = []
        targets = []
        ids = []

        has_labels = "label" in df.columns

        for _, row in df.iterrows():
            spec = compute_spectrogram(row["filepath"])
            specs.append(spec)

            if has_labels:
                targets.append(row["label"])

            ids.append(row["clip"])

        # Convert to numpy arrays
        # Stack specs: (N, 1, F, T)
        data = np.stack(specs).astype(np.float32)

        # Save cache
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        np.save(data_path, data)

        if has_labels:
            targets = np.array(targets).astype(np.float32)  # BCE takes float
            np.save(labels_path, targets)
            return data, targets, None
        else:
            ids = np.array(ids)
            np.save(ids_path, ids)
            return data, None, ids


class WhaleDataset(Dataset):
    def __init__(self, data, targets=None, ids=None, transform=False):
        """
        data: (N, 1, F, T) numpy array of spectrograms
        targets: (N,) numpy array of labels (optional)
        ids: (N,) numpy array of clip names (optional)
        transform: bool, whether to apply SpecAugment
        """
        self.data = data
        self.targets = targets
        self.ids = ids
        self.transform = transform

        # Define Augmentations
        # SpecAugment
        self.time_masking = T.TimeMasking(time_mask_param=Config.TIME_MASK_PARAM)
        self.freq_masking = T.FrequencyMasking(freq_mask_param=Config.FREQ_MASK_PARAM)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get spectrogram
        spec = torch.from_numpy(self.data[idx])  # (1, F, T)

        # Apply Augmentation if training
        if self.transform:
            # SpecAugment expects (channel, freq, time) or (freq, time)
            # Our spec is (1, F, T). Transforms work on tensor.
            spec = self.freq_masking(spec)
            spec = self.time_masking(spec)

        if self.targets is not None:
            label = torch.tensor(self.targets[idx], dtype=torch.float32)
            return spec, label
        else:
            clip_id = self.ids[idx]
            return spec, clip_id


def get_dataloaders(load_cached_data=True):
    """
    Prepare DataLoaders for Train, Val, and Test.
    """
    # 1. Process/Load Data
    train_data, train_labels, _ = process_and_cache_data(
        Config.TRAIN_CSV, "train", load_cached_data
    )
    val_data, val_labels, _ = process_and_cache_data(
        Config.VAL_CSV, "val", load_cached_data
    )
    test_data, _, test_ids = process_and_cache_data(
        Config.TEST_CSV, "test", load_cached_data
    )

    # 2. Create Datasets
    # Train: Apply Augmentation
    train_dataset = WhaleDataset(train_data, targets=train_labels, transform=True)

    # Val: No Augmentation
    val_dataset = WhaleDataset(val_data, targets=val_labels, transform=False)

    # Test: No Augmentation
    test_dataset = WhaleDataset(test_data, ids=test_ids, transform=False)

    # 3. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    return train_loader, val_loader, test_loader
