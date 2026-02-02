import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from library.config import Config

# Ensure reproducible behavior
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)


def get_transforms(mode="train"):
    """
    Returns the augmentation pipeline for the given mode.
    """
    if mode == "train" and Config.SPEC_AUGMENT:
        return torch.nn.Sequential(
            torchaudio.transforms.FrequencyMasking(
                freq_mask_param=Config.FREQ_MASK_PARAM
            ),
            torchaudio.transforms.TimeMasking(time_mask_param=Config.TIME_MASK_PARAM),
        )
    return None


class WhaleDataset(Dataset):
    def __init__(self, data, labels=None, clips=None, transform=None):
        """
        Args:
            data (np.ndarray): Shape (N, C, F, T)
            labels (np.ndarray, optional): Shape (N,)
            clips (np.ndarray, optional): Shape (N,) - filenames for submission
            transform (callable, optional): Augmentation function
        """
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).float() if labels is not None else None
        self.clips = clips
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Data is already (C, F, T)
        x = self.data[idx]

        # Apply transforms (SpecAugment) if provided
        if self.transform:
            x = self.transform(x)

        if self.labels is not None:
            y = self.labels[idx]
            return x, y
        else:
            clip = self.clips[idx]
            return x, clip


def preprocess_waveform(waveform, sample_rate):
    """
    Pads/crops waveform to fixed duration, computes MelSpectrogram,
    converts to DB, and applies Instance Normalization.
    """
    # 1. Fix Duration (Pad or Crop)
    target_length = int(Config.SAMPLE_RATE * Config.DURATION)

    # Resample if necessary (though analysis says all are 2000Hz)
    if sample_rate != Config.SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=Config.SAMPLE_RATE
        )
        waveform = resampler(waveform)

    # Ensure mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    current_length = waveform.shape[1]
    if current_length < target_length:
        padding = target_length - current_length
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    elif current_length > target_length:
        waveform = waveform[:, :target_length]

    # 2. Compute Mel Spectrogram
    # Note: We instantiate transforms here for clarity, but for speed in a loop
    # it's often better to instantiate once. However, this function is used in data prep
    # which happens once (cached), so this is acceptable.
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=Config.SAMPLE_RATE,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH,
        n_mels=Config.N_MELS,
        f_min=Config.FMIN,
        f_max=Config.FMAX,
        normalized=False,  # We do instance norm later
    )

    spec = mel_transform(waveform)

    # 3. Convert to DB
    db_transform = torchaudio.transforms.AmplitudeToDB()
    spec = db_transform(spec)

    # 4. Instance Normalization (Zero-Mean, Unit-Variance per clip)
    if Config.NORMALIZE_INSTANCE:
        mean = spec.mean()
        std = spec.std()
        # Avoid division by zero
        spec = (spec - mean) / (std + 1e-6)

    return spec


def load_dataset_data(mode, load_cached_data=True):
    """
    Loads data for a specific mode ('train', 'val', 'test').
    Uses caching to avoid re-processing audio files.
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    data_path = os.path.join(cache_dir, f"{mode}_data.npy")
    labels_path = os.path.join(cache_dir, f"{mode}_labels.npy")
    clips_path = os.path.join(cache_dir, f"{mode}_clips.npy")

    # Check cache
    has_cache = os.path.exists(data_path)
    if mode == "test":
        has_cache = has_cache and os.path.exists(clips_path)
    else:
        has_cache = has_cache and os.path.exists(labels_path)

    if load_cached_data and has_cache:
        print(f"Loading cached {mode} data from {cache_dir}...")
        data = np.load(data_path)
        if mode == "test":
            clips = np.load(clips_path)
            return data, None, clips
        else:
            labels = np.load(labels_path)
            return data, labels, None

    print(f"Processing {mode} data from scratch...")

    # Select Metadata File
    if mode == "train":
        df = pd.read_csv(Config.TRAIN_CSV)
    elif mode == "val":
        df = pd.read_csv(Config.VAL_CSV)
    else:
        df = pd.read_csv(Config.TEST_CSV)

    data_list = []
    labels_list = []
    clips_list = []

    # Pre-instantiate transforms for efficiency
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=Config.SAMPLE_RATE,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH,
        n_mels=Config.N_MELS,
        f_min=Config.FMIN,
        f_max=Config.FMAX,
        normalized=False,
    )
    db_transform = torchaudio.transforms.AmplitudeToDB()
    resampler = None  # Will instantiate if needed

    total_files = len(df)

    for idx, row in df.iterrows():
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        try:
            waveform, sr = torchaudio.load(file_path)

            # --- Inline Preprocessing for Speed ---

            # Resample
            if sr != Config.SAMPLE_RATE:
                if resampler is None or resampler.orig_freq != sr:
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=sr, new_freq=Config.SAMPLE_RATE
                    )
                waveform = resampler(waveform)

            # Mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Pad/Crop
            target_length = int(Config.SAMPLE_RATE * Config.DURATION)
            current_length = waveform.shape[1]
            if current_length < target_length:
                padding = target_length - current_length
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            elif current_length > target_length:
                waveform = waveform[:, :target_length]

            # Spectrogram
            spec = mel_transform(waveform)
            spec = db_transform(spec)

            # Instance Norm
            if Config.NORMALIZE_INSTANCE:
                m = spec.mean()
                s = spec.std()
                spec = (spec - m) / (s + 1e-6)

            data_list.append(spec.numpy())

            if mode != "test":
                labels_list.append(row["label"])
            else:
                clips_list.append(row["clip"])

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            # In a real scenario, we might skip or pad with zeros.
            # For this competition, we assume data integrity or skip.
            # To keep array aligned, we must append something or drop the row.
            # We'll skip and warn.
            continue

    # Convert to numpy arrays
    data_arr = np.stack(data_list)  # Shape: (N, 1, F, T)

    # Save to cache
    np.save(data_path, data_arr)
    print(f"Saved {mode} data to {data_path}")

    if mode == "test":
        clips_arr = np.array(clips_list)
        np.save(clips_path, clips_arr)
        return data_arr, None, clips_arr
    else:
        labels_arr = np.array(labels_list)
        np.save(labels_path, labels_arr)
        return data_arr, labels_arr, None


def get_dataloaders(load_cached_data=True):
    """
    Creates and returns dataloaders for train, val, and test sets.
    """
    # 1. Load Data
    train_data, train_labels, _ = load_dataset_data("train", load_cached_data)
    val_data, val_labels, _ = load_dataset_data("val", load_cached_data)
    test_data, _, test_clips = load_dataset_data("test", load_cached_data)

    # 2. Create Datasets
    train_dataset = WhaleDataset(
        train_data, train_labels, transform=get_transforms("train")
    )

    val_dataset = WhaleDataset(val_data, val_labels, transform=get_transforms("val"))

    test_dataset = WhaleDataset(
        test_data, clips=test_clips, transform=get_transforms("test")
    )

    # 3. Create Sampler for Class Imbalance (Train only)
    # Calculate weights: Inverse of class frequency
    class_counts = np.bincount(train_labels.astype(int))
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_labels.astype(int)]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_labels), replacement=True
    )

    # 4. Create Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=sampler,  # Mutually exclusive with shuffle
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader
