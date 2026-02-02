import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from library.config import Config
from library.features import cache_dataset

# Ensure deterministic behavior for augmentations where possible
torch.manual_seed(Config.SEED)


def apply_spec_augment(
    spec, num_freq_masks=2, freq_mask_param=10, num_time_masks=2, time_mask_param=15
):
    """
    Applies SpecAugment (Frequency and Time Masking) to the spectrogram.

    Args:
        spec (torch.Tensor): Input tensor of shape (Channels, Freq, Time).
        num_freq_masks (int): Number of frequency masks to apply.
        freq_mask_param (int): Maximum width of frequency mask.
        num_time_masks (int): Number of time masks to apply.
        time_mask_param (int): Maximum width of time mask.

    Returns:
        torch.Tensor: Augmented tensor.
    """
    # Work on a clone to avoid modifying the cached tensor in memory
    if isinstance(spec, np.ndarray):
        spec = torch.from_numpy(spec)

    aug_spec = spec.clone()
    C, F, T = aug_spec.shape

    # Fill value is the minimum value in the spectrogram (background/silence level)
    fill_value = aug_spec.min()

    # --- Frequency Masking ---
    for _ in range(num_freq_masks):
        # Inclusive bounds for width: [0, freq_mask_param]
        f = int(torch.randint(0, freq_mask_param + 1, (1,)).item())
        f = min(f, F)  # Clamp to max frequency
        if f == 0:
            continue

        # Start position: [0, F - f]
        f0 = int(torch.randint(0, F - f + 1, (1,)).item())

        # Apply mask across all channels
        aug_spec[:, f0 : f0 + f, :] = fill_value

    # --- Time Masking ---
    # Constraint: Strictly limit time masks to < 20% of the duration
    max_allowed_time_mask = int(0.2 * T) - 1
    # Use the stricter of the provided param or the safety limit
    actual_time_mask_param = min(time_mask_param, max_allowed_time_mask)

    if actual_time_mask_param > 0:
        for _ in range(num_time_masks):
            # Inclusive bounds for width
            t = int(torch.randint(0, actual_time_mask_param + 1, (1,)).item())
            t = min(t, T)
            if t == 0:
                continue

            # Start position: [0, T - t]
            t0 = int(torch.randint(0, T - t + 1, (1,)).item())

            # Apply mask across all channels
            aug_spec[:, :, t0 : t0 + t] = fill_value

    return aug_spec


class CachedSpeechDataset(Dataset):
    """
    Dataset that loads pre-computed spectrograms from .npy files.
    """

    def __init__(self, df, augment=False):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'cache_path' and 'label'.
            augment (bool): Whether to apply SpecAugment.
        """
        self.df = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cache_path = row["cache_path"]
        label_str = row["label"]

        # Load the pre-computed spectrogram (3, 64, 101)
        # We assume cache_dataset has already been run and files exist.
        try:
            spec_np = np.load(cache_path)
            spec_tensor = torch.from_numpy(spec_np).float()
        except Exception as e:
            # Fallback for corruption or missing file (should not happen if cache_dataset ran)
            # Return a silent tensor
            spec_tensor = torch.zeros((3, Config.N_MELS, 101), dtype=torch.float32)

        # Apply Augmentation if requested
        if self.augment:
            spec_tensor = apply_spec_augment(spec_tensor)

        # Convert label string to ID
        # 'unknown' label in test set maps to the 'unknown' ID
        label_id = Config.LABEL2ID.get(label_str, Config.LABEL2ID["unknown"])

        return spec_tensor, torch.tensor(label_id, dtype=torch.long)


def get_dataloaders():
    """
    Prepares DataLoaders for train, validation, and test sets.
    Handles caching of features and weighted sampling for class imbalance.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 1. Load Metadata
    train_csv = os.path.join(Config.METADATA_DIR, "train.csv")
    val_csv = os.path.join(Config.METADATA_DIR, "val.csv")
    test_csv = os.path.join(Config.METADATA_DIR, "test.csv")

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_test = pd.read_csv(test_csv)

    # 2. Debug / Subsampling
    if Config.DEBUG or (Config.MAX_TRAIN_SAMPLES is not None):
        n = Config.MAX_TRAIN_SAMPLES if Config.MAX_TRAIN_SAMPLES else 200
        print(f"DEBUG Mode: Subsampling datasets to {n} samples.")
        df_train = df_train.head(n)
        df_val = df_val.head(n)
        df_test = df_test.head(n)

    # 3. Cache Features (Compute and Save .npy files)
    print("Processing and caching Training data...")
    df_train = cache_dataset(df_train, Config.CACHE_DIR)

    print("Processing and caching Validation data...")
    df_val = cache_dataset(df_val, Config.CACHE_DIR)

    print("Processing and caching Test data...")
    df_test = cache_dataset(df_test, Config.CACHE_DIR)

    # 4. Configure WeightedRandomSampler for Training
    # Calculate class weights to handle imbalance (e.g., 'unknown' class dominance)
    label_counts = df_train["label"].value_counts()
    class_weights = {label: 1.0 / count for label, count in label_counts.items()}

    # Assign a weight to each sample based on its label
    sample_weights = [class_weights[label] for label in df_train["label"]]

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(df_train), replacement=True
    )

    # 5. Instantiate Datasets
    train_dataset = CachedSpeechDataset(df_train, augment=True)
    val_dataset = CachedSpeechDataset(df_val, augment=False)
    test_dataset = CachedSpeechDataset(df_test, augment=False)

    # 6. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=sampler,  # Use sampler instead of shuffle=True
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
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
