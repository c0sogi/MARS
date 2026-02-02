import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class SpecAugment:
    """
    Applies Time and Frequency Masking to spectrograms.
    """

    def __init__(self, time_mask=30, freq_mask=40):
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=time_mask)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=freq_mask
        )

    def __call__(self, spec):
        # spec shape: (Channels, Freq, Time)
        # TimeMasking and FrequencyMasking expect (..., Freq, Time)
        return self.freq_mask(self.time_mask(spec))


class WhaleDataset(Dataset):
    """
    PyTorch Dataset for Right Whale Detection.
    """

    def __init__(self, images, targets=None, transform=None, is_test=False):
        """
        Args:
            images (np.ndarray): Array of spectrograms with shape (N, 1, F, T).
            targets (np.ndarray or list): Array of labels or clip names.
            transform (callable, optional): Augmentation function.
            is_test (bool): If True, returns (image, clip_name). If False, returns (image, label).
        """
        self.images = images
        self.targets = targets
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load data as tensor
        img = torch.tensor(self.images[idx], dtype=torch.float32)

        # Apply Augmentations (e.g., SpecAugment)
        if self.transform:
            img = self.transform(img)

        if self.is_test:
            # Return image and clip_name for submission generation
            return img, self.targets[idx]
        else:
            # Return image and label (float for BCE loss compatibility)
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return img, target


def compute_spectrogram(audio, sr, config):
    """
    Converts raw audio to a normalized Log-Mel Spectrogram.
    """
    # 1. Pad or Truncate to fixed duration
    target_samples = int(config.DURATION * sr)
    current_samples = len(audio)

    if current_samples < target_samples:
        pad_width = target_samples - current_samples
        audio = np.pad(audio, (0, pad_width), mode="constant")
    else:
        audio = audio[:target_samples]

    audio_tensor = torch.tensor(audio, dtype=torch.float32)

    # 2. Compute Mel Spectrogram
    mel_spec_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS,
        f_min=config.F_MIN,
        f_max=config.F_MAX,
    )
    spec = mel_spec_transform(audio_tensor)

    # 3. Convert to Log Scale (dB)
    db_transform = torchaudio.transforms.AmplitudeToDB()
    spec = db_transform(spec)

    # 4. Instance-level Min-Max Normalization
    min_val = spec.min()
    max_val = spec.max()

    # Avoid division by zero
    if max_val - min_val > 1e-6:
        spec = (spec - min_val) / (max_val - min_val)
    else:
        spec = torch.zeros_like(spec)

    # Add channel dimension: (1, F, T)
    spec = spec.unsqueeze(0)

    return spec.numpy()


def process_and_cache_data(split, df, config):
    """
    Reads audio files, computes spectrograms, and returns arrays.
    """
    print(f"Processing {split} dataset ({len(df)} samples)...")

    specs = []
    labels = []
    names = []

    # Debug mode: limit samples
    if config.DEBUG:
        df = df.head(config.DEBUG_SAMPLE_SIZE)
        print(f"Debug mode: processing only {len(df)} samples.")

    for _, row in df.iterrows():
        file_path = os.path.join(config.INPUT_ROOT, row["file_path"])

        try:
            # Load Audio
            audio, sr = sf.read(file_path)

            # Generate Spectrogram
            spec = compute_spectrogram(audio, sr, config)
            specs.append(spec)

            # Store Metadata
            if "label" in row:
                labels.append(row["label"])
            if "clip_name" in row:
                names.append(row["clip_name"])

        except Exception as e:
            # Handle corrupt files by creating a zero-tensor placeholder
            print(f"Error processing {file_path}: {e}")
            # Calculate expected time dimension
            n_frames = int(config.DURATION * config.SAMPLE_RATE / config.HOP_LENGTH) + 1
            dummy_spec = np.zeros((1, config.N_MELS, n_frames), dtype=np.float32)
            specs.append(dummy_spec)

            if "label" in row:
                labels.append(0)  # Default to noise
            if "clip_name" in row:
                names.append(row["clip_name"])

    # Convert to numpy arrays
    specs_arr = np.array(specs, dtype=np.float32)
    labels_arr = np.array(labels, dtype=np.float32) if labels else None
    names_arr = np.array(names) if names else None

    return specs_arr, labels_arr, names_arr


def get_data_for_split(split, load_cached_data=True, config=Config):
    """
    Retrieves data for a split, handling caching logic.
    """
    # Define cache filename
    cache_filename = f"{split}_debug.npz" if config.DEBUG else f"{split}_data.npz"
    cache_path = os.path.join(config.CACHE_DIR, cache_filename)

    # 1. Try Loading from Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading {split} data from cache: {cache_path}")
            data = np.load(cache_path, allow_pickle=True)

            specs = data["specs"]
            # Handle potential None values stored in npz
            labels = (
                data["labels"]
                if "labels" in data.files and data["labels"].ndim > 0
                else None
            )
            names = (
                data["names"]
                if "names" in data.files and data["names"].ndim > 0
                else None
            )

            return specs, labels, names
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing.")

    # 2. Process from Scratch
    if split == "train":
        df = pd.read_csv(config.TRAIN_METADATA_PATH)
    elif split == "val":
        df = pd.read_csv(config.VAL_METADATA_PATH)
    elif split == "test":
        df = pd.read_csv(config.TEST_METADATA_PATH)
    else:
        raise ValueError(f"Invalid split: {split}")

    specs, labels, names = process_and_cache_data(split, df, config)

    # 3. Save to Cache
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    np.savez(cache_path, specs=specs, labels=labels, names=names)
    print(f"Saved {split} data to cache.")

    return specs, labels, names


def get_dataloaders(load_cached_data=True, config=Config, pseudo_labels=None):
    """
    Constructs DataLoaders for the pipeline.

    Args:
        load_cached_data (bool): Whether to use cached data.
        config (Config): Configuration object.
        pseudo_labels (dict, optional): Dictionary mapping clip_name -> probability.
                                        Used to augment training data for Student training.

    Returns:
        train_loader, val_loader, test_loader
    """

    # --- Load Data ---
    train_specs, train_labels, _ = get_data_for_split("train", load_cached_data, config)
    val_specs, val_labels, _ = get_data_for_split("val", load_cached_data, config)
    test_specs, _, test_names = get_data_for_split("test", load_cached_data, config)

    # --- Handle Pseudo-Labeling ---
    if pseudo_labels is not None:
        print(f"Applying Pseudo-Labels. Original Train Size: {len(train_specs)}")

        # Identify test samples that have pseudo-labels
        pseudo_indices = []
        pseudo_targets = []

        for idx, name in enumerate(test_names):
            if name in pseudo_labels:
                pseudo_indices.append(idx)
                pseudo_targets.append(pseudo_labels[name])

        if pseudo_indices:
            # Extract pseudo-labeled data
            extra_specs = test_specs[pseudo_indices]
            extra_labels = np.array(pseudo_targets, dtype=np.float32)

            # Concatenate with original training data
            train_specs = np.concatenate([train_specs, extra_specs], axis=0)
            train_labels = np.concatenate([train_labels, extra_labels], axis=0)

            print(
                f"Augmented Train Size: {len(train_specs)} (Added {len(extra_specs)} pseudo-labeled samples)"
            )
        else:
            print("Warning: No matching pseudo-labels found in test set.")

    # --- Create Datasets ---
    # Training dataset gets Augmentation
    train_transform = SpecAugment(
        time_mask=config.SPEC_AUG_TIME_MASK, freq_mask=config.SPEC_AUG_FREQ_MASK
    )

    train_dataset = WhaleDataset(
        train_specs, train_labels, transform=train_transform, is_test=False
    )

    val_dataset = WhaleDataset(val_specs, val_labels, transform=None, is_test=False)

    test_dataset = WhaleDataset(test_specs, test_names, transform=None, is_test=True)

    # --- Create Loaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Essential for Batch Norm stability
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
