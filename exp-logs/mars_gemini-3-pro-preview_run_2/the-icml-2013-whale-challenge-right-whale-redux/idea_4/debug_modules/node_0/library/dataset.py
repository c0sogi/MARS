import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from library import config, utils

# Ensure reproducibility
utils.set_seed(config.SEED)


class WhaleDataset(Dataset):
    """
    Custom Dataset for Right Whale Detection.
    Stores pre-computed Log-Mel Spectrograms in memory for fast access.
    """

    def __init__(self, data, labels=None, clips=None, transform=None):
        """
        Args:
            data (np.ndarray): Shape (N, 1, F, T) - Precomputed spectrograms
            labels (np.ndarray, optional): Shape (N,) - Ground truth labels
            clips (np.ndarray, optional): Shape (N,) - Clip filenames for submission
            transform (callable, optional): Augmentations to apply
        """
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long() if labels is not None else None
        self.clips = clips
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve spectrogram: Shape (1, F, T)
        spec = self.data[idx]

        # Apply augmentations (e.g., SpecAugment) if provided
        # Note: SpecAugment expects (C, F, T) or (F, T)
        if self.transform:
            spec = self.transform(spec)

        # Instance-wise Standardization (Zero Mean, Unit Variance)
        # Critical for neural network stability with varying audio levels
        mean = spec.mean()
        std = spec.std()
        if std > 1e-6:
            spec = (spec - mean) / std
        else:
            spec = spec - mean

        # Return appropriate tuple based on mode (Train/Val vs Test)
        if self.labels is not None:
            return spec, self.labels[idx]
        elif self.clips is not None:
            return spec, self.clips[idx]
        else:
            return spec


def get_transforms(mode="train"):
    """
    Returns the transformation pipeline for the given mode.
    """
    if mode == "train":
        # SpecAugment: Randomly masks time and frequency bands
        # Helps the model be robust to partial signal occlusions
        return torch.nn.Sequential(
            torchaudio.transforms.TimeMasking(time_mask_param=10),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=20),
        )
    else:
        # No augmentation for validation/test
        return None


def compute_spectrogram(waveform, sample_rate):
    """
    Converts a raw waveform into a Log-Mel Spectrogram.
    Aligns with the physics-based parameters in config.
    """
    # Ensure waveform is (1, Time)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    # Pad or truncate to fixed length (2.0s = 4000 samples at 2000Hz)
    # This ensures consistent input dimensions for the CNN
    target_len = int(2.0 * sample_rate)
    current_len = waveform.shape[1]

    if current_len < target_len:
        pad_amount = target_len - current_len
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
    elif current_len > target_len:
        waveform = waveform[:, :target_len]

    # Compute Mel Spectrogram
    mel_spec_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS,
        f_min=config.FMIN,
        f_max=config.FMAX,
        center=True,
    )
    spec = mel_spec_transform(waveform)

    # Convert to Log Scale (dB)
    # top_db=80 restricts the dynamic range to 80dB, typical for audio
    to_db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80.0)
    log_mel_spec = to_db_transform(spec)

    return log_mel_spec


def process_data(
    metadata_path, data_cache, label_cache=None, clip_cache=None, load_cached=True
):
    """
    Loads metadata, processes audio files into spectrograms, and manages caching.
    """
    # 1. Try to load from cache
    has_data = os.path.exists(data_cache)
    has_label = (label_cache is None) or os.path.exists(label_cache)
    has_clip = (clip_cache is None) or os.path.exists(clip_cache)

    if load_cached and has_data and has_label and has_clip:
        print(f"Loading cached data from {data_cache}...")
        data = np.load(data_cache)
        labels = np.load(label_cache) if label_cache else None
        clips = np.load(clip_cache) if clip_cache else None
        return data, labels, clips

    # 2. Process from scratch
    print(f"Processing data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    data_list = []
    label_list = []
    clip_list = []

    # Ensure working directory exists for cache
    os.makedirs(os.path.dirname(data_cache), exist_ok=True)

    for idx, row in df.iterrows():
        # Construct full file path
        file_path = os.path.join(config.INPUT_DIR, row["file_path"])

        try:
            # Load Audio (soundfile is robust for .aif)
            wav, sr = sf.read(file_path)

            # Convert to Tensor
            wav_tensor = torch.from_numpy(wav).float()

            # Compute Spectrogram
            spec = compute_spectrogram(wav_tensor, config.SAMPLE_RATE)

            # Append to list (convert to numpy to save memory overhead of tensors in list)
            data_list.append(spec.numpy())

            if "label" in row:
                label_list.append(row["label"])

            if "clip" in row:
                clip_list.append(row["clip"])

        except Exception as e:
            print(f"Warning: Error processing {file_path}: {e}")
            continue

    # Stack into single arrays
    # Data shape: (N, 1, F, T)
    data_array = np.stack(data_list)

    # Save to cache
    np.save(data_cache, data_array)
    print(f"Saved processed data to {data_cache}")

    labels_array = None
    if label_list:
        labels_array = np.array(label_list)
        if label_cache:
            np.save(label_cache, labels_array)

    clips_array = None
    if clip_list:
        clips_array = np.array(clip_list)
        if clip_cache:
            np.save(clip_cache, clips_array)

    return data_array, labels_array, clips_array


def get_dataloaders(load_cached_data=True):
    """
    Generates DataLoaders for Train, Validation, and Test sets.
    Handles caching, sampling, and batching.
    """
    # --- Train Data ---
    train_data, train_labels, _ = process_data(
        config.TRAIN_METADATA,
        config.TRAIN_DATA_CACHE,
        config.TRAIN_LABELS_CACHE,
        load_cached=load_cached_data,
    )

    # --- Validation Data ---
    val_data, val_labels, _ = process_data(
        config.VAL_METADATA,
        config.VAL_DATA_CACHE,
        config.VAL_LABELS_CACHE,
        load_cached=load_cached_data,
    )

    # --- Test Data ---
    test_data, _, test_clips = process_data(
        config.TEST_METADATA,
        config.TEST_DATA_CACHE,
        clip_cache=config.TEST_CLIPS_CACHE,
        load_cached=load_cached_data,
    )

    # --- Datasets ---
    train_dataset = WhaleDataset(
        train_data, train_labels, transform=get_transforms("train")
    )
    val_dataset = WhaleDataset(val_data, val_labels, transform=get_transforms("val"))
    test_dataset = WhaleDataset(
        test_data, clips=test_clips, transform=get_transforms("test")
    )

    # --- Weighted Sampler for Class Imbalance ---
    # Calculate weights inversely proportional to class frequency
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    # Assign weight to each sample
    sample_weights = class_weights[train_labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    # --- DataLoaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=sampler,  # Handles shuffling and balancing
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
