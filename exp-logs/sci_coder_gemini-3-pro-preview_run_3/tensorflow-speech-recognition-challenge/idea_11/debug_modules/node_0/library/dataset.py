import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from library.config import train_cfg, audio_cfg, path_cfg

# --- Label Configuration ---
# The order must be fixed to ensure consistency.
# First 10 are the target commands, followed by silence and unknown.
TARGET_LABELS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
AUX_LABELS = ["silence", "unknown"]
ALL_LABELS = TARGET_LABELS + AUX_LABELS

LABEL2IDX = {label: idx for idx, label in enumerate(ALL_LABELS)}
IDX2LABEL = {idx: label for idx, label in enumerate(ALL_LABELS)}


class SpecAugment:
    """
    Applies Frequency and Time Masking to the spectrogram.
    Applied to all channels identically to preserve multi-resolution alignment.
    """

    def __init__(self, freq_mask_param=10, time_mask_param=None, time_mask_ratio=0.2):
        self.freq_mask_param = freq_mask_param
        self.time_mask_ratio = time_mask_ratio
        # If time_mask_param is not set explicitly, it is calculated per sample based on length
        self.time_mask_param = time_mask_param

    def __call__(self, spec):
        """
        Args:
            spec (Tensor): Shape (Channels, Freq, Time) -> (3, 64, T)
        Returns:
            Tensor: Masked spectrogram.
        """
        # spec is (C, F, T)
        C, F, T = spec.shape

        # Calculate min value for filling
        fill_value = spec.min().item()

        # --- Frequency Masking ---
        # Mask width
        f = random.randint(0, self.freq_mask_param)
        # Mask start (inclusive bounds)
        f0 = random.randint(0, F - f)

        # Apply mask to all channels
        spec[:, f0 : f0 + f, :] = fill_value

        # --- Time Masking ---
        # Determine max allowed time mask length
        max_mask_len = int(T * self.time_mask_ratio)

        # Use configured param if valid, else use ratio constraint
        if self.time_mask_param is not None:
            t_param = min(self.time_mask_param, max_mask_len)
        else:
            t_param = max_mask_len

        if t_param > 0:
            t = random.randint(0, t_param)
            t0 = random.randint(0, T - t)

            # Apply mask to all channels
            spec[:, :, t0 : t0 + t] = fill_value

        return spec


class SpeechCommandDataset(Dataset):
    def __init__(self, dataframe, transform=False):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame with 'cache_path' and 'label'.
            transform (bool): Whether to apply SpecAugment.
        """
        self.df = dataframe
        self.transform = transform

        # Initialize augmenter if needed
        if self.transform:
            # Parameters can be tuned. Freq mask 15 bins, Time mask dynamic (<20%)
            self.augmenter = SpecAugment(freq_mask_param=15, time_mask_ratio=0.15)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cache_path = row["cache_path"]

        # Load spectrogram: Shape (3, 64, T)
        try:
            spec_np = np.load(cache_path)
            spec = torch.from_numpy(spec_np).float()
        except Exception as e:
            # Fallback for corrupted/missing cache (should not happen if preprocessing ran)
            # Create a dummy silent spectrogram
            # T is approx 101 for 1s audio with hop 160
            spec = torch.ones(3, 64, 101) * -80.0  # approx min db

        # Apply Augmentation
        if self.transform:
            spec = self.augmenter(spec)

        # Get Label
        label_str = row["label"]
        # For test set, label might be 'unknown' placeholder, which maps to 11
        label_idx = LABEL2IDX.get(label_str, LABEL2IDX["unknown"])

        return spec, torch.tensor(label_idx, dtype=torch.long)


def get_dataloaders(train_csv_path, val_csv_path, test_csv_path):
    """
    Creates DataLoaders for Train, Validation, and Test sets.
    Implements WeightedRandomSampler for the training set.
    """
    # 1. Load DataFrames
    df_train = pd.read_csv(train_csv_path)
    df_val = pd.read_csv(val_csv_path)
    df_test = pd.read_csv(test_csv_path)

    # 2. Prepare Training Dataset & Sampler
    train_dataset = SpeechCommandDataset(df_train, transform=True)

    # Calculate weights for WeightedRandomSampler
    # Count occurrences of each label
    label_counts = df_train["label"].value_counts()

    # Calculate weight for each sample: 1.0 / count of its class
    # This balances the probability of picking any class
    sample_weights = []
    for label in df_train["label"]:
        count = label_counts.get(label, 0)
        if count > 0:
            weight = 1.0 / count
        else:
            weight = 0.0
        sample_weights.append(weight)

    sample_weights = torch.tensor(sample_weights, dtype=torch.double)

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        sampler=sampler,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # 3. Prepare Validation Dataset
    val_dataset = SpeechCommandDataset(df_val, transform=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
    )

    # 4. Prepare Test Dataset
    test_dataset = SpeechCommandDataset(df_test, transform=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
