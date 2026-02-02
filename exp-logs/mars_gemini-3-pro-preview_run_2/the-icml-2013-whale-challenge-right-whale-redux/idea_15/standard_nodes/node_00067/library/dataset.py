import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from library.config import Config
from library.utils import seed_everything

# Ensure reproducibility
seed_everything(Config.SEED)


class WhaleDataset(Dataset):
    """
    PyTorch Dataset for Right Whale Call Detection.
    Handles in-memory tensors and on-the-fly SpecAugment.
    """

    def __init__(self, data, targets, mode="train", transform=None):
        """
        Args:
            data (Tensor): Pre-processed audio spectrograms (N, 1, F, T).
            targets (Tensor or list): Labels for train/val, clip names for test.
            mode (str): 'train', 'val', or 'test'.
            transform (nn.Module, optional): Augmentation pipeline.
        """
        self.data = data
        self.targets = targets
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Data is already (C, F, T)
        img = self.data[idx]

        # Apply Augmentation (SpecAugment) only in training mode
        if self.mode == "train" and self.transform:
            img = self.transform(img)

        target = self.targets[idx]

        # Return format depends on mode
        if self.mode == "test":
            # For test, target is the clip filename
            return img, target
        else:
            # For train/val, target is the label (0 or 1)
            # Ensure label is a tensor
            if isinstance(target, torch.Tensor):
                return img, target.float()
            return img, torch.tensor(target, dtype=torch.float32)


def get_audio_transforms():
    """
    Creates the deterministic MelSpectrogram pipeline.
    Note: Instance normalization is applied manually after this.
    """
    mel_spec = T.MelSpectrogram(
        sample_rate=Config.SR,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH,
        n_mels=Config.N_MELS,
        power=Config.POWER,
        normalized=Config.NORMALIZE_MELS,
        f_min=Config.FMIN,
        f_max=Config.FMAX,
    )

    # AmplitudeToDB with top_db clamping
    amp_to_db = T.AmplitudeToDB(stype="power", top_db=Config.TOP_DB)

    return torch.nn.Sequential(mel_spec, amp_to_db)


def get_augmentations():
    """
    Creates SpecAugment pipeline for training.
    """
    return torch.nn.Sequential(
        T.TimeMasking(time_mask_param=10), T.FrequencyMasking(freq_mask_param=10)
    )


def preprocess_audio(file_path, transform_pipeline, target_length=4000):
    """
    Loads, pads, transforms, and normalizes a single audio file.
    """
    full_path = os.path.join(Config.INPUT_ROOT, file_path)

    # Load audio
    try:
        waveform, sr = torchaudio.load(full_path)
    except Exception as e:
        # Fallback for empty/corrupt files (should not happen based on metadata check)
        print(f"Error loading {full_path}: {e}")
        waveform = torch.zeros(1, target_length)
        sr = Config.SR

    # Resample if necessary (though dataset is uniform 2kHz)
    if sr != Config.SR:
        resampler = T.Resample(sr, Config.SR)
        waveform = resampler(waveform)

    # Pad or Crop to fixed length (2.0s = 4000 samples)
    # This ensures consistent output dimensions for batching (128x63)
    c, t = waveform.shape
    if t < target_length:
        padding = target_length - t
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    elif t > target_length:
        waveform = waveform[:, :target_length]

    # Compute MelSpectrogram -> dB
    spec = transform_pipeline(waveform)

    # Instance-wise Zero-Mean Unit-Variance Standardization
    # Normalize per sample to handle varying noise levels
    mean = spec.mean()
    std = spec.std()
    spec = (spec - mean) / (std + 1e-6)

    return spec


def process_and_cache_data(df, cache_prefix, load_cached_data=True):
    """
    Handles caching logic: Load from disk if available, else compute and save.
    """
    data_path = os.path.join(Config.WORKING_DIR, f"{cache_prefix}_data.npy")
    targets_path = os.path.join(Config.WORKING_DIR, f"{cache_prefix}_targets.npy")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(data_path) and os.path.exists(targets_path):
        print(f"Loading cached {cache_prefix} data from {Config.WORKING_DIR}...")
        data_np = np.load(data_path)
        targets_np = np.load(targets_path, allow_pickle=True)

        # Convert back to Tensor
        data_tensor = torch.from_numpy(data_np)
        # Targets might be strings (clips) or ints (labels)
        if "test" in cache_prefix:
            targets = targets_np.tolist()  # Keep clips as list of strings
        else:
            targets = torch.from_numpy(targets_np).long()

        return data_tensor, targets

    # 2. Compute from scratch
    print(f"Processing {cache_prefix} data (Cache miss or force reload)...")

    transform_pipeline = get_audio_transforms()
    data_list = []
    targets_list = []

    # Determine target column
    is_test = "test" in cache_prefix
    target_col = "clip" if is_test else "label"

    for idx, row in df.iterrows():
        # Process Audio
        spec = preprocess_audio(row["file_path"], transform_pipeline)
        data_list.append(spec)

        # Store Target
        targets_list.append(row[target_col])

        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1}/{len(df)} files...")

    # Stack into Tensor
    data_tensor = torch.stack(data_list)

    # Prepare targets for saving
    if is_test:
        targets_np = np.array(targets_list)
    else:
        targets_tensor = torch.tensor(targets_list, dtype=torch.long)
        targets_np = targets_tensor.numpy()

    # Save to disk (numpy format)
    np.save(data_path, data_tensor.numpy())
    np.save(targets_path, targets_np)
    print(f"Saved processed {cache_prefix} data to {Config.WORKING_DIR}")

    if is_test:
        return data_tensor, targets_list
    else:
        return data_tensor, torch.tensor(targets_list, dtype=torch.long)


def get_dataloaders(debug=False, load_cached_data=True):
    """
    Main factory function to create DataLoaders.

    Args:
        debug (bool): If True, uses a small subset of data.
        load_cached_data (bool): If True, attempts to load pre-processed .npy files.

    Returns:
        dict: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """

    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Debug Subset
    if debug:
        print(f"DEBUG MODE: Using {Config.DEBUG_SUBSET_SIZE} samples per split.")
        train_df = train_df.head(Config.DEBUG_SUBSET_SIZE)
        val_df = val_df.head(Config.DEBUG_SUBSET_SIZE)
        test_df = test_df.head(Config.DEBUG_SUBSET_SIZE)

    # Process/Load Data
    # Note: We use unique prefixes for debug mode to avoid overwriting full cache
    prefix_suffix = "_debug" if debug else ""

    train_data, train_labels = process_and_cache_data(
        train_df, f"train{prefix_suffix}", load_cached_data
    )
    val_data, val_labels = process_and_cache_data(
        val_df, f"val{prefix_suffix}", load_cached_data
    )
    test_data, test_clips = process_and_cache_data(
        test_df, f"test{prefix_suffix}", load_cached_data
    )

    # Create Datasets
    # Augmentation only for Train
    augment_pipeline = get_augmentations()

    train_dataset = WhaleDataset(
        train_data, train_labels, mode="train", transform=augment_pipeline
    )
    val_dataset = WhaleDataset(val_data, val_labels, mode="val", transform=None)
    test_dataset = WhaleDataset(test_data, test_clips, mode="test", transform=None)

    # Weighted Sampler for Training (Class Imbalance)
    # Calculate weights
    class_counts = torch.bincount(train_labels)
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=sampler,  # Mutually exclusive with shuffle
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

    return {"train": train_loader, "val": val_loader, "test": test_loader}
