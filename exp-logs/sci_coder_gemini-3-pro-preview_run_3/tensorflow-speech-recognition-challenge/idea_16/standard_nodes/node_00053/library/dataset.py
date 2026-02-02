import os
import numpy as np
import pandas as pd
import torch
import soundfile as sf
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from library.config import Config
from library.utils import set_seed

# Ensure reproducibility
set_seed(Config.SEED)


class SpeechCommandsDataset(Dataset):
    def __init__(self, data, targets, phase="train"):
        """
        Args:
            data (np.ndarray): Audio data of shape (N, samples).
            targets (np.ndarray): Label indices of shape (N,).
            phase (str): 'train', 'val', or 'test'.
        """
        self.data = data
        self.targets = targets
        self.phase = phase

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get waveform and ensure float32
        waveform = self.data[idx].astype(np.float32)

        # Convert to tensor
        waveform_tensor = torch.from_numpy(waveform)

        # Get label
        label = self.targets[idx]
        label_tensor = torch.tensor(label, dtype=torch.long)

        return waveform_tensor, label_tensor


def load_and_pad_audio(filepath, target_length=16000):
    """
    Reads an audio file and pads/crops it to the target length.
    """
    try:
        # Load audio
        wav, sr = sf.read(filepath)

        # Ensure mono
        if len(wav.shape) > 1:
            wav = wav[:, 0]

        # Pad or Crop
        if len(wav) < target_length:
            pad_width = target_length - len(wav)
            # Pad at the end
            wav = np.pad(wav, (0, pad_width), mode="constant")
        elif len(wav) > target_length:
            # Crop from center
            start = (len(wav) - target_length) // 2
            wav = wav[start : start + target_length]

        return wav
    except Exception as e:
        # In case of error (e.g. corrupt file), return silence
        return np.zeros(target_length, dtype=np.float32)


def process_and_cache_data(df, cache_prefix, load_cached_data=True):
    """
    Loads audio data from dataframe paths, processes them, and caches to disk.

    Args:
        df (pd.DataFrame): Dataframe containing 'filepath' and 'label'.
        cache_prefix (str): Prefix for cache files (e.g., 'train').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        data (np.ndarray): (N, samples)
        targets (np.ndarray): (N,)
    """
    cache_dir = os.path.join(Config.WORKING_DIR, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    data_path = os.path.join(cache_dir, f"{cache_prefix}_data.npy")
    targets_path = os.path.join(cache_dir, f"{cache_prefix}_targets.npy")

    # 1. Try to load from cache
    if load_cached_data:
        if os.path.exists(data_path) and os.path.exists(targets_path):
            print(f"Loading {cache_prefix} data from cache...")
            try:
                data = np.load(data_path)
                targets = np.load(targets_path)
                return data, targets
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")
        else:
            print(f"Cache not found for {cache_prefix}. Processing...")

    # 2. Process from scratch
    print(f"Processing {len(df)} files for {cache_prefix}...")

    num_samples = len(df)
    audio_len = Config.NUM_SAMPLES

    # Pre-allocate arrays
    data = np.zeros((num_samples, audio_len), dtype=np.float32)
    targets = np.zeros(num_samples, dtype=np.int64)

    # Iterate through dataframe
    for i, row in df.iterrows():
        # Construct full path
        full_path = os.path.join(Config.INPUT_ROOT, row["filepath"])

        # Load audio
        wav = load_and_pad_audio(full_path, target_length=audio_len)
        data[i] = wav

        # Map label
        label_str = row["label"]
        # Use .get() to handle potential unexpected labels safely, defaulting to unknown
        label_id = Config.LABEL2ID.get(label_str, Config.LABEL2ID[Config.UNKNOWN_LABEL])
        targets[i] = label_id

    # 3. Save to cache
    print(f"Saving {cache_prefix} to cache...")
    np.save(data_path, data)
    np.save(targets_path, targets)

    return data, targets


def get_dataloaders(load_cached_data=True):
    """
    Prepares DataLoaders for train, val, and test splits.

    Args:
        load_cached_data (bool): Whether to use cached .npy files.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Handle Debug Mode
    # We use a unique prefix for debug mode to avoid overwriting full cache
    prefix_suffix = "_debug" if Config.DEBUG else ""

    if Config.DEBUG:
        train_df = train_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
        val_df = val_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
        test_df = test_df.iloc[: Config.DEBUG_SAMPLE_SIZE]

    # Process Data (Load or Compute)
    train_data, train_targets = process_and_cache_data(
        train_df, f"train{prefix_suffix}", load_cached_data
    )
    val_data, val_targets = process_and_cache_data(
        val_df, f"val{prefix_suffix}", load_cached_data
    )
    test_data, test_targets = process_and_cache_data(
        test_df, f"test{prefix_suffix}", load_cached_data
    )

    # Create Datasets
    train_dataset = SpeechCommandsDataset(train_data, train_targets, phase="train")
    val_dataset = SpeechCommandsDataset(val_data, val_targets, phase="val")
    test_dataset = SpeechCommandsDataset(test_data, test_targets, phase="test")

    # Create WeightedRandomSampler for Training to handle class imbalance
    # Calculate class counts
    class_counts = np.bincount(train_targets, minlength=Config.NUM_CLASSES)

    # Avoid division by zero
    class_counts = np.maximum(class_counts, 1)

    # Weights: inverse frequency
    class_weights = 1.0 / class_counts

    # Assign weight to each sample based on its label
    sample_weights = class_weights[train_targets]

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=sampler,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
