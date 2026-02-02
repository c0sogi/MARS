import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.feature_engineering import process_data
from library.utils import get_logger

# Initialize logger
logger = get_logger("dataset")


class VolcanoCNNDataset(Dataset):
    """
    PyTorch Dataset for the Contrast-Normalized Vision Model.

    Handles:
    1. Loading pre-computed spectrograms.
    2. Applying Instance Standardization (per-sample normalization).
    3. Applying Log-Scaling to targets (if available).
    """

    def __init__(self, spectrograms, targets=None):
        """
        Args:
            spectrograms (np.ndarray): Tensor of shape (N, 10, Freq, Time).
            targets (np.ndarray, optional): Array of targets (time_to_eruption).
        """
        self.spectrograms = spectrograms
        self.targets = targets

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, idx):
        # Load spectrogram: Shape (10, 128, T)
        spec = torch.tensor(self.spectrograms[idx], dtype=torch.float32)

        # Apply Instance Standardization
        # Normalize each spectrogram instance independently: (X - mean) / std
        if Config.INSTANCE_STANDARDIZATION:
            mean = spec.mean()
            std = spec.std()
            # Add epsilon to prevent division by zero
            spec = (spec - mean) / (std + 1e-6)

        # Handle Targets
        if self.targets is not None:
            target_val = self.targets[idx]

            # Apply Log-Scaling to target: log1p(y)
            if Config.TARGET_LOG_SCALE:
                target_val = np.log1p(target_val)

            target = torch.tensor(target_val, dtype=torch.float32)
            return spec, target
        else:
            # Inference mode (if targets not provided)
            return spec


class VolcanoTabularBuilder:
    """
    Builder class to orchestrate data loading and feature engineering.

    Uses library.feature_engineering.process_data to:
    1. Load metadata.
    2. Extract robust tabular features (MFCCs, Stats).
    3. Generate Log-Mel Spectrograms.
    4. Handle caching to disk.
    """

    def __init__(self):
        self.logger = get_logger("VolcanoTabularBuilder")

    def get_data(self, split="train", load_cache=True):
        """
        Retrieves processed data for a specific split.

        Args:
            split (str): One of 'train', 'val', 'test'.
            load_cache (bool): Whether to attempt loading from cache.

        Returns:
            tuple: (df_tabular, X_spectrograms, y_targets)
        """
        # Map split to metadata file
        if split == "train":
            meta_path = os.path.join(Config.METADATA_DIR, "train.csv")
        elif split == "val":
            meta_path = os.path.join(Config.METADATA_DIR, "val.csv")
        elif split == "test":
            meta_path = os.path.join(Config.METADATA_DIR, "test.csv")
        else:
            raise ValueError(
                f"Invalid split name: {split}. Must be 'train', 'val', or 'test'."
            )

        self.logger.info(f"Building dataset for split: {split}")

        # Delegate to the library function which handles parallel processing and caching
        df_tabular, X_spectrograms, y_targets = process_data(
            metadata_path=meta_path, dataset_name=split, load_cached_data=load_cache
        )

        self.logger.info(
            f"Data ready for {split}. "
            f"Tabular: {df_tabular.shape}, "
            f"Spectrograms: {X_spectrograms.shape}, "
            f"Targets: {y_targets.shape}"
        )

        return df_tabular, X_spectrograms, y_targets
