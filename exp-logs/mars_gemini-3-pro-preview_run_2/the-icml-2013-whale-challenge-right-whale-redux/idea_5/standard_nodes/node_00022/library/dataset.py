import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchaudio import transforms as T
import torchaudio.functional as F
from library.config import Config

# Ensure reproducible behavior
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)


class AudioPreprocessor:
    """
    Handles the conversion of raw audio files to 3-channel tensors
    (Log-Mel, Delta, Delta-Delta).
    """

    def __init__(self):
        self.mel_transform = T.MelSpectrogram(
            sample_rate=Config.SR,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            f_min=Config.FMIN,
            f_max=Config.FMAX,
            normalized=True,
        )
        self.db_transform = T.AmplitudeToDB(top_db=80)
        self.target_length = int(Config.SR * Config.DURATION)

    def process_file(self, file_path):
        # Load audio
        try:
            waveform, sr = torchaudio.load(file_path)
        except Exception as e:
            # Fallback for corrupted or missing files (should not happen based on metadata check)
            # Create a silent waveform
            waveform = torch.zeros(1, self.target_length)
            sr = Config.SR

        # Resample if necessary
        if sr != Config.SR:
            resampler = T.Resample(sr, Config.SR)
            waveform = resampler(waveform)

        # Ensure mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Pad or Truncate to fixed length
        if waveform.shape[1] < self.target_length:
            padding = self.target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif waveform.shape[1] > self.target_length:
            waveform = waveform[:, : self.target_length]

        # Generate Mel Spectrogram
        mel_spec = self.mel_transform(waveform)
        log_mel_spec = self.db_transform(mel_spec)  # Shape: (1, n_mels, time)

        # Cite solution_lesson_node_00021: Prefer raw single-channel spectrograms
        # Stack channels: (1, n_mels, time)
        image = log_mel_spec

        # Instance Normalization per channel
        # Mean/Std over (Freq, Time)
        mean = image.mean(dim=(1, 2), keepdim=True)
        std = image.std(dim=(1, 2), keepdim=True)
        image = (image - mean) / (std + 1e-6)

        return image


def load_dataset_data(mode, load_cached_data=True):
    """
    Loads data for a specific mode (train, val, test).
    Handles caching logic using .npy files.
    """
    # Determine paths based on mode
    if mode == "train":
        csv_path = Config.TRAIN_CSV
        data_cache = Config.TRAIN_DATA_CACHE
        label_cache = Config.TRAIN_LABELS_CACHE
    elif mode == "val":
        csv_path = Config.VAL_CSV
        data_cache = Config.VAL_DATA_CACHE
        label_cache = Config.VAL_LABELS_CACHE
    elif mode == "test":
        csv_path = Config.TEST_CSV
        data_cache = Config.TEST_DATA_CACHE
        label_cache = Config.TEST_CLIPS_CACHE  # Stores clip names instead of labels
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Ensure working directory exists
    os.makedirs(os.path.dirname(data_cache), exist_ok=True)

    # Check cache
    if load_cached_data and os.path.exists(data_cache) and os.path.exists(label_cache):
        print(f"Loading {mode} data from cache...")
        try:
            data = np.load(data_cache)
            labels = np.load(
                label_cache, allow_pickle=True
            )  # allow_pickle needed for strings in test
            return data, labels
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    print(f"Processing {mode} data from scratch...")

    # Load metadata
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Debugging: subset if configured
    if Config.DEBUG:
        df = df.iloc[: Config.DEBUG_SAMPLES]

    preprocessor = AudioPreprocessor()
    processed_data = []
    processed_labels = []

    for idx, row in df.iterrows():
        file_path = os.path.join(Config.INPUT_ROOT, row["file_path"])

        # Process audio
        tensor = preprocessor.process_file(file_path)
        processed_data.append(tensor.numpy())

        # Store label or clip name
        if mode == "test":
            processed_labels.append(row["clip"])
        else:
            processed_labels.append(row["label"])

    # Convert to numpy arrays
    data_array = np.stack(processed_data)  # Shape: (N, 3, F, T)
    labels_array = np.array(processed_labels)

    # Save to cache
    np.save(data_cache, data_array)
    np.save(label_cache, labels_array)
    print(f"Saved {mode} data to cache.")

    return data_array, labels_array


class WhaleDataset(Dataset):
    def __init__(self, data, labels, training=False):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = labels  # Can be tensor (int) or list/array (string for test)
        self.training = training

        # Augmentations
        if self.training:
            # SpecAugment
            # TimeMasking and FrequencyMasking apply to the tensor (..., F, T)
            self.time_masking = T.TimeMasking(time_mask_param=10)
            self.freq_masking = T.FrequencyMasking(freq_mask_param=20)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]

        if self.training:
            # Apply SpecAugment
            x = self.freq_masking(x)
            x = self.time_masking(x)

        y = self.labels[idx]

        # If labels are integers (train/val), convert to tensor
        if not isinstance(y, str):
            y = torch.tensor(
                y, dtype=torch.float32
            )  # BCEWithLogitsLoss expects float target

        return x, y


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for train, val, and test sets.
    """
    # Load Data
    train_data, train_labels = load_dataset_data("train", load_cached_data)
    val_data, val_labels = load_dataset_data("val", load_cached_data)
    test_data, test_clips = load_dataset_data("test", load_cached_data)

    # Create Datasets
    train_dataset = WhaleDataset(train_data, train_labels, training=True)
    val_dataset = WhaleDataset(val_data, val_labels, training=False)
    test_dataset = WhaleDataset(test_data, test_clips, training=False)

    # Create WeightedRandomSampler for training to handle class imbalance
    # train_labels is numpy array of 0s and 1s
    class_counts = np.bincount(train_labels.astype(int))
    # Handle potential zero division if a class is missing (unlikely)
    class_weights = 1.0 / (class_counts + 1e-6)

    # Assign weight to each sample
    sample_weights = class_weights[train_labels.astype(int)]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=sampler,  # Mutually exclusive with shuffle
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
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
