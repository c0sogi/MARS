import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import library.config as config
from library.config import (
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
)
from library.feature_engineering import process_tabular_data, generate_spectrogram
from library.utils import seed_everything

# Set random seed for reproducibility in data loading/shuffling
seed_everything(SEED)


class VolcanoDataset(Dataset):
    """
    PyTorch Dataset for loading seismic sensor data and generating spectrograms on-the-fly.
    Designed for Stream B (CNN) of the ensemble.
    """

    def __init__(self, metadata_path, is_test=False, debug=False):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file (train/val/test).
            is_test (bool): Whether this is the test set (no target variable available).
            debug (bool): If True, limits the dataset to a small subset for debugging purposes.
        """
        self.metadata_path = metadata_path
        self.is_test = is_test
        self.debug = debug

        # Load metadata
        self.df = pd.read_csv(metadata_path)

        # Handle debugging mode: slice the dataframe
        if self.debug:
            self.df = self.df.iloc[: config.DEBUG_SAMPLE_SIZE].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns:
            spectrogram (Tensor): Shape (C, H, W) -> (10, 224, 224)
            target (Tensor): Float32 scalar (time_to_eruption) or 0.0 for test
            segment_id (int): ID of the segment
        """
        row = self.df.iloc[idx]
        file_path = row["file_path"]
        segment_id = int(row["segment_id"])

        # Generate Log-Mel Spectrogram using the library function
        # Returns Tensor of shape (C, H, W)
        spectrogram = generate_spectrogram(file_path)

        if self.is_test:
            # For test set, return dummy target
            target = torch.tensor(0.0, dtype=torch.float32)
        else:
            # For train/val, return actual target
            target_val = row["time_to_eruption"]
            target = torch.tensor(target_val, dtype=torch.float32)

        return spectrogram, target, segment_id


def prepare_tabular_dataset(load_cached_data=True):
    """
    Generates or loads tabular features for Train, Validation, and Test sets.
    This function handles the 'Stream A' data preparation for LightGBM.

    Args:
        load_cached_data (bool): If True, attempts to load from Parquet cache in WORKING_DIR.
                                 If False or cache missing, re-computes features.

    Returns:
        tuple: (train_df, val_df, test_df) - Pandas DataFrames containing features and targets.
    """
    # Train Set
    train_df = process_tabular_data(
        metadata_path=TRAIN_META_PATH,
        save_filename="train_features.parquet",
        load_cached_data=load_cached_data,
    )

    # Validation Set
    val_df = process_tabular_data(
        metadata_path=VAL_META_PATH,
        save_filename="val_features.parquet",
        load_cached_data=load_cached_data,
    )

    # Test Set
    test_df = process_tabular_data(
        metadata_path=TEST_META_PATH,
        save_filename="test_features.parquet",
        load_cached_data=load_cached_data,
    )

    return train_df, val_df, test_df


def get_spectrogram_loaders(batch_size=BATCH_SIZE, debug=False):
    """
    Creates PyTorch DataLoaders for the CNN stream.

    Args:
        batch_size (int): Batch size for the dataloaders.
        debug (bool): If True, uses a small subset of data for all sets.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Create Datasets
    train_dataset = VolcanoDataset(TRAIN_META_PATH, is_test=False, debug=debug)
    val_dataset = VolcanoDataset(VAL_META_PATH, is_test=False, debug=debug)
    test_dataset = VolcanoDataset(TEST_META_PATH, is_test=True, debug=debug)

    # Create DataLoaders
    # Pin memory helps with transfer to GPU
    # Drop last for training to avoid unstable batch norm stats on small last batches
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True if not debug else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
