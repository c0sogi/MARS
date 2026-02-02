import torch
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from library.config import TrainConfig, AudioConfig
from library.preprocess import get_feature_path

# Fixed label ordering for consistent mapping
LABELS = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
    "silence",
    "unknown",
]
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}
IDX2LABEL = {i: l for i, l in enumerate(LABELS)}


class SpecAugment(torch.nn.Module):
    """
    Custom SpecAugment implementation to handle 3-channel inputs and
    specific fill values.
    """

    def __init__(self, freq_mask_param, time_mask_param, min_val):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.min_val = min_val

    def forward(self, spec):
        """
        Args:
            spec (torch.Tensor): Shape (channels, n_mels, time)
        Returns:
            torch.Tensor: Masked spectrogram
        """
        # spec is (C, F, T)
        C, F, T = spec.shape

        # --- Frequency Masking ---
        # Generate mask size with inclusive bounds [0, F_param]
        f_mask_len = int(torch.randint(0, self.freq_mask_param + 1, (1,)).item())
        if f_mask_len > 0 and f_mask_len < F:
            f_start = int(torch.randint(0, F - f_mask_len + 1, (1,)).item())
            # Apply min_val to the mask region across all channels
            spec[:, f_start : f_start + f_mask_len, :] = self.min_val

        # --- Time Masking ---
        # Generate mask size with inclusive bounds [0, T_param]
        t_mask_len = int(torch.randint(0, self.time_mask_param + 1, (1,)).item())
        if t_mask_len > 0 and t_mask_len < T:
            t_start = int(torch.randint(0, T - t_mask_len + 1, (1,)).item())
            # Apply min_val to the mask region across all channels
            spec[:, :, t_start : t_start + t_mask_len] = self.min_val

        return spec


class CachedSpeechDataset(Dataset):
    """
    Dataset that loads pre-computed multi-resolution spectrograms from disk.
    """

    def __init__(self, metadata_df, transform=None, is_test=False):
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]

        # Load cached feature
        # get_feature_path is imported from library.preprocess
        # It assumes TrainConfig.cache_dir is set correctly
        npy_path = get_feature_path(row["filepath"])

        try:
            # Load (3, n_mels, time)
            features = np.load(npy_path)
            features = torch.from_numpy(features).float()
        except Exception as e:
            # Fallback for missing/corrupt files (should be rare if cache is built)
            # Return a silent tensor
            features = (
                torch.ones((3, AudioConfig.n_mels, 101)) * TrainConfig.spec_aug_min_val
            )

        # Apply Augmentation
        if self.transform:
            features = self.transform(features)

        if self.is_test:
            return features, row["filepath"]
        else:
            label_str = row["label"]
            label_idx = LABEL2IDX.get(label_str, LABEL2IDX["unknown"])
            return features, torch.tensor(label_idx, dtype=torch.long)


def get_balanced_dataloader(
    metadata_path, batch_size, is_training=True, subset_size=None
):
    """
    Creates a DataLoader. If is_training=True, uses WeightedRandomSampler
    to balance classes.
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    # Debugging subset
    if subset_size is not None:
        df = df.sample(
            n=min(len(df), subset_size), random_state=TrainConfig.seed
        ).reset_index(drop=True)

    # Setup Transforms
    transform = None
    if is_training:
        transform = SpecAugment(
            freq_mask_param=TrainConfig.spec_aug_freq_mask_param,
            time_mask_param=TrainConfig.spec_aug_time_mask_param,
            min_val=TrainConfig.spec_aug_min_val,
        )

    dataset = CachedSpeechDataset(df, transform=transform, is_test=False)

    if is_training and TrainConfig.use_weighted_sampler:
        # Calculate class weights
        label_counts = df["label"].value_counts()
        # Weight = 1 / count
        class_weights = {label: 1.0 / count for label, count in label_counts.items()}

        # Assign weight to each sample
        sample_weights = df["label"].map(class_weights).fillna(0).values
        sample_weights = torch.from_numpy(sample_weights).double()

        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=TrainConfig.num_workers,
            pin_memory=True,
        )
    else:
        # Validation or Training without sampler
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_training,
            num_workers=TrainConfig.num_workers,
            pin_memory=True,
        )

    return loader


def get_test_dataloader(metadata_path, batch_size):
    """
    Creates a DataLoader for the test set (no labels, no shuffling).
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)
    dataset = CachedSpeechDataset(df, transform=None, is_test=True)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=TrainConfig.num_workers,
        pin_memory=True,
    )

    return loader
