import os
import hashlib
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from joblib import Parallel, delayed
from library.config import Config
from library.audio_processor import load_audio, generate_multires_spectrogram

# Ensure cache directory exists
os.makedirs(Config.CACHE_DIR, exist_ok=True)


class SpecAugment:
    """
    Applies SpecAugment (Frequency and Time Masking) to the spectrogram.
    Applied to all channels identically to preserve inter-channel relationships.
    """

    def __init__(
        self,
        freq_mask_param=Config.FREQ_MASK_PARAM,
        time_mask_param=Config.TIME_MASK_PARAM,
    ):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param

    def __call__(self, spec):
        # spec shape: (C, F, T).
        # We work on a copy to avoid modifying the cached array in memory if referenced elsewhere
        augmented_spec = spec.copy()

        C, F, T = augmented_spec.shape

        # Determine fill value (min of the whole spec to represent silence/absence)
        fill_value = augmented_spec.min()

        # 1. Frequency Masking
        f = np.random.randint(0, self.freq_mask_param + 1)
        if F - f > 0:
            f0 = np.random.randint(0, F - f + 1)
            augmented_spec[:, f0 : f0 + f, :] = fill_value

        # 2. Time Masking
        t = np.random.randint(0, self.time_mask_param + 1)
        if T - t > 0:
            t0 = np.random.randint(0, T - t + 1)
            augmented_spec[:, :, t0 : t0 + t] = fill_value

        return augmented_spec


def get_cache_filename(filepath):
    """Generates a unique cache filename based on the input filepath."""
    hasher = hashlib.md5()
    hasher.update(filepath.encode("utf-8"))
    return f"{hasher.hexdigest()}.npy"


def process_single_file(row, cache_dir, load_cached_data):
    """
    Worker function to process a single audio file.
    Returns (cache_path, label_id) for train/val, or just cache_path for test.
    """
    filepath = row["filepath"]
    label = row["label"]

    # Map label to ID if it exists in the row, otherwise default to unknown
    label_id = Config.LABEL2ID.get(label, Config.LABEL2ID["unknown"])

    cache_filename = get_cache_filename(filepath)
    cache_path = os.path.join(cache_dir, cache_filename)

    # Check if exists and we want to load it
    if load_cached_data and os.path.exists(cache_path):
        return cache_path, label_id

    # Otherwise compute and save
    try:
        waveform = load_audio(filepath)
        spec = generate_multires_spectrogram(waveform)
        np.save(cache_path, spec)
        return cache_path, label_id
    except Exception:
        # Return None to indicate failure
        return None, None


class CachedSpeechDataset(Dataset):
    def __init__(self, data_list, transform=None):
        """
        Args:
            data_list: List of tuples (cache_path, label_id)
            transform: Function/Transform to apply on the spectrogram
        """
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        cache_path, label_id = self.data_list[idx]

        try:
            spec = np.load(cache_path)
        except Exception:
            # Fallback for corrupt files (creates a zero tensor of approx correct shape)
            # Shape: (3, 64, 101) based on 16000 samples and hop 160 + centering
            spec = np.zeros((Config.IN_CHANNELS, Config.N_MELS, 101), dtype=np.float32)

        if self.transform:
            spec = self.transform(spec)

        spec_tensor = torch.from_numpy(spec).float()
        label_tensor = torch.tensor(label_id, dtype=torch.long)

        return spec_tensor, label_tensor


class TestDataset(Dataset):
    def __init__(self, data_list):
        """
        Args:
            data_list: List of tuples (cache_path, original_filename)
        """
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        cache_path, fname = self.data_list[idx]
        try:
            spec = np.load(cache_path)
        except Exception:
            spec = np.zeros((Config.IN_CHANNELS, Config.N_MELS, 101), dtype=np.float32)

        spec_tensor = torch.from_numpy(spec).float()
        return spec_tensor, fname


def cache_dataset(metadata_path, cache_dir, load_cached_data=True, debug=False):
    """
    Iterates metadata, caches spectrograms, and returns list of data items.
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    if debug:
        df = df.head(Config.DEBUG_SAMPLE_SIZE)

    rows = df.to_dict("records")

    # Parallel processing for speed
    # joblib requires n_jobs >= 1, whereas PyTorch uses 0 for main process
    n_jobs = Config.NUM_WORKERS if Config.NUM_WORKERS > 0 else 1
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_single_file)(row, cache_dir, load_cached_data) for row in rows
    )

    # Filter out failures
    valid_results = [r for r in results if r[0] is not None]

    return valid_results


def get_dataloaders(debug=False, load_cached_data=True):
    """
    Prepares DataLoaders for Train and Validation.
    """
    # --- Train Data ---
    train_data = cache_dataset(
        Config.TRAIN_METADATA,
        Config.CACHE_DIR,
        load_cached_data=load_cached_data,
        debug=debug,
    )

    # Calculate weights for WeightedRandomSampler
    train_labels = [item[1] for item in train_data]
    class_counts = np.bincount(train_labels, minlength=Config.NUM_CLASSES)

    # Avoid division by zero
    class_counts = np.maximum(class_counts, 1)
    class_weights = 1.0 / class_counts

    # Assign weight to each sample
    sample_weights = [class_weights[label] for label in train_labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_data), replacement=True
    )

    train_dataset = CachedSpeechDataset(train_data, transform=SpecAugment())

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=sampler,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # --- Validation Data ---
    val_data = cache_dataset(
        Config.VAL_METADATA,
        Config.CACHE_DIR,
        load_cached_data=load_cached_data,
        debug=debug,
    )

    val_dataset = CachedSpeechDataset(val_data, transform=None)

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_dataloader(debug=False, load_cached_data=True):
    """
    Prepares DataLoader for Test set.
    """
    if not os.path.exists(Config.TEST_METADATA):
        raise FileNotFoundError(f"Metadata file not found: {Config.TEST_METADATA}")

    df = pd.read_csv(Config.TEST_METADATA)
    if debug:
        df = df.head(Config.DEBUG_SAMPLE_SIZE)

    rows = df.to_dict("records")

    # Process files
    n_jobs = Config.NUM_WORKERS if Config.NUM_WORKERS > 0 else 1
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_single_file)(row, Config.CACHE_DIR, load_cached_data)
        for row in rows
    )

    # Reconstruct list of (cache_path, fname)
    test_data_list = []
    for i, (cache_path, _) in enumerate(results):
        if cache_path is not None:
            fname = os.path.basename(rows[i]["filepath"])
            test_data_list.append((cache_path, fname))

    dataset = TestDataset(test_data_list)

    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return loader
