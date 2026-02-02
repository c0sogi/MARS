import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from library.config import Config
from library.utils import seed_everything

# Ensure deterministic behavior
seed_everything(Config.SEED)


class WhaleDataset(Dataset):
    """
    Custom Dataset for Right Whale Call Detection.
    Holds pre-processed Mel Spectrograms in memory for efficiency.
    """

    def __init__(self, data, labels=None, clips=None, transform=None):
        """
        Args:
            data (np.ndarray): Array of shape (N, 1, F, T) containing spectrograms.
            labels (np.ndarray, optional): Array of shape (N,) containing targets.
            clips (np.ndarray, optional): Array of shape (N,) containing filenames (for test).
            transform (callable, optional): Augmentation pipeline.
        """
        self.data = data
        self.labels = labels
        self.clips = clips
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve spectrogram
        x = self.data[idx]

        # Convert to tensor (ensure float32)
        x = torch.from_numpy(x).float()

        # Apply augmentations (e.g., SpecAugment) if provided
        if self.transform:
            x = self.transform(x)

        # Return data and label (or clip name for test)
        if self.labels is not None:
            y = torch.tensor(self.labels[idx], dtype=torch.float32)
            return x, y
        else:
            clip = self.clips[idx]
            return x, clip


def get_transforms(mode="train"):
    """
    Returns the augmentation pipeline.
    """
    if mode == "train":
        # SpecAugment: Masking in Time and Frequency domains
        # Applied after normalization, masking with 0 (mean) is appropriate.
        return nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=20),
            torchaudio.transforms.TimeMasking(time_mask_param=10),
        )
    else:
        return None


def compute_spectrogram(filepath):
    """
    Reads audio, pads/crops, computes Mel Spectrogram, converts to dB,
    and applies Instance Standardization.
    """
    # 1. Load Audio
    try:
        audio, sr = sf.read(filepath)
    except Exception as e:
        # Fallback for corrupt files (should be rare)
        audio = np.zeros(Config.NUM_SAMPLES)
        sr = Config.SAMPLE_RATE

    # 2. Handle Channels (Convert to Mono)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # 3. Pad or Crop to Fixed Length (Native Resolution Strategy)
    if len(audio) < Config.NUM_SAMPLES:
        pad_width = Config.NUM_SAMPLES - len(audio)
        audio = np.pad(audio, (0, pad_width), mode="constant")
    else:
        audio = audio[: Config.NUM_SAMPLES]

    # Convert to Tensor for torchaudio
    audio_tensor = torch.from_numpy(audio).float()

    # 4. Compute Mel Spectrogram
    # Using "Golden Recipe" parameters
    mel_spec_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=Config.SAMPLE_RATE,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH,
        n_mels=Config.N_MELS,
        f_min=Config.FMIN,
        f_max=Config.FMAX,
        normalized=Config.MEL_NORMALIZED,  # False, preserving Pink noise tilt
    )

    # Shape: (n_mels, time)
    spec = mel_spec_transform(audio_tensor)

    # 5. Convert to Log Scale (dB)
    db_transform = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
    spec_db = db_transform(spec)

    # 6. Instance Standardization (Zero-Mean, Unit-Variance per clip)
    mean = spec_db.mean()
    std = spec_db.std()
    spec_norm = (spec_db - mean) / (std + 1e-6)

    # Add channel dimension: (1, F, T)
    spec_norm = spec_norm.unsqueeze(0)

    return spec_norm.numpy().astype(np.float32)


def process_and_cache_data(split, load_cached_data=True):
    """
    Loads processed data from disk if available; otherwise computes and caches it.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (data, labels, clips)
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache file paths
    data_path = os.path.join(Config.WORKING_DIR, f"{split}_data.npy")
    labels_path = os.path.join(Config.WORKING_DIR, f"{split}_labels.npy")
    clips_path = os.path.join(Config.WORKING_DIR, f"{split}_clips.npy")

    # Check existence
    cache_exists = os.path.exists(data_path)
    if split != "test":
        cache_exists = cache_exists and os.path.exists(labels_path)
    else:
        cache_exists = cache_exists and os.path.exists(clips_path)

    # 1. Try Loading from Cache
    if load_cached_data and cache_exists:
        # print(f"Loading {split} data from cache...")
        data = np.load(data_path, allow_pickle=True)
        if split != "test":
            labels = np.load(labels_path, allow_pickle=True)
            return data, labels, None
        else:
            clips = np.load(clips_path, allow_pickle=True)
            return data, None, clips

    # 2. Compute from Scratch
    print(f"Processing {split} data from scratch...")

    # Load Metadata
    if split == "train":
        df = pd.read_csv(Config.TRAIN_CSV)
    elif split == "val":
        df = pd.read_csv(Config.VAL_CSV)
    elif split == "test":
        df = pd.read_csv(Config.TEST_CSV)
    else:
        raise ValueError(f"Unknown split: {split}")

    data_list = []
    labels_list = []
    clips_list = []

    # Iterate and Process
    for _, row in df.iterrows():
        file_path = os.path.join(Config.INPUT_ROOT, row["file_path"])
        spec = compute_spectrogram(file_path)
        data_list.append(spec)

        if split != "test":
            labels_list.append(row["label"])
        else:
            clips_list.append(row["clip"])

    # Stack into single array
    data = np.stack(data_list)  # Shape: (N, 1, F, T)

    # Save to Cache
    np.save(data_path, data)

    if split != "test":
        labels = np.array(labels_list, dtype=np.int64)
        np.save(labels_path, labels)
        return data, labels, None
    else:
        clips = np.array(clips_list, dtype=object)
        np.save(clips_path, clips)
        return data, None, clips


def get_data_loader(split, batch_size=None, shuffle=None, load_cached_data=True):
    """
    Creates and returns a DataLoader for the specified split.
    Handles caching, sampling, and transforms.
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE

    # Load Data (Cached or Fresh)
    data, labels, clips = process_and_cache_data(
        split, load_cached_data=load_cached_data
    )

    # Define Transforms
    transform = get_transforms("train") if split == "train" else None

    # Instantiate Dataset
    dataset = WhaleDataset(data, labels=labels, clips=clips, transform=transform)

    # Configure Sampler / Shuffle
    sampler = None

    # For Training, use WeightedRandomSampler to handle class imbalance
    if split == "train":
        if shuffle is None:  # Default to True logic via Sampler
            # Calculate weights for balancing
            class_counts = np.bincount(labels)
            class_weights = 1.0 / class_counts
            sample_weights = class_weights[labels]

            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            shuffle = False  # Sampler and shuffle are mutually exclusive
    else:
        # Default shuffle to False for val/test unless specified
        if shuffle is None:
            shuffle = False

    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=(split == "train"),  # Drop last incomplete batch only during training
    )

    return loader
