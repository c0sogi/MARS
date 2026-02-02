import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.preprocessing import DataPreprocessor


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ManufacturingDataset(Dataset):
    """
    PyTorch Dataset for the Manufacturing Control Data.
    Separates continuous and categorical features for the DCNv2 model.
    """

    def __init__(self, df, is_test=False):
        self.is_test = is_test

        # Define feature column groups matching Preprocessing logic
        # Continuous: Base f_00-f_30 (excl f_27) + engineered 'unique_character_count'
        self.cont_cols = Config.BASE_CONT_COLS + ["unique_character_count"]
        # Categorical: Decomposed characters char_0 to char_9
        self.cat_cols = [f"char_{i}" for i in range(Config.STR_LEN)]

        # Extract features as Numpy arrays for efficient indexing
        # Ensure float32 for continuous inputs
        self.cont_features = df[self.cont_cols].values.astype(np.float32)

        # Ensure int64 (long) for embedding lookups
        self.cat_features = df[self.cat_cols].values.astype(np.int64)

        # Extract targets or IDs
        if not self.is_test:
            # Training/Validation: Load targets
            self.targets = df[Config.TARGET_COL].values.astype(np.float32)
            self.ids = None
        else:
            # Test: Load IDs for submission
            self.targets = None
            self.ids = df[Config.ID_COL].values

    def __len__(self):
        return len(self.cont_features)

    def __getitem__(self, idx):
        # Construct the sample dictionary
        item = {
            "continuous": torch.tensor(self.cont_features[idx], dtype=torch.float32),
            "categorical": torch.tensor(self.cat_features[idx], dtype=torch.long),
        }

        if not self.is_test:
            # Return target with shape (1,) for BCE loss compatibility
            item["target"] = torch.tensor(
                self.targets[idx], dtype=torch.float32
            ).unsqueeze(0)
        else:
            # Return ID for submission mapping
            item["id"] = self.ids[idx]

        return item


def get_dataloaders(load_cached_data=True, debug=False):
    """
    Orchestrates the data pipeline:
    1. Sets seeds.
    2. Uses DataPreprocessor to load/cache/process data.
    3. Wraps data in ManufacturingDataset.
    4. Returns PyTorch DataLoaders.

    Args:
        load_cached_data (bool): Whether to attempt loading from parquet cache.
        debug (bool): Whether to use a small subset of data for debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 1. Ensure Reproducibility
    set_seed(Config.SEED)

    # 2. Preprocess Data
    # The DataPreprocessor handles the caching logic internally (checking paths, saving parquet)
    preprocessor = DataPreprocessor()
    train_df, val_df, test_df = preprocessor.process_data(
        load_cached_data=load_cached_data, debug=debug
    )

    # 3. Create Datasets
    train_dataset = ManufacturingDataset(train_df, is_test=False)
    val_dataset = ManufacturingDataset(val_df, is_test=False)
    test_dataset = ManufacturingDataset(test_df, is_test=True)

    # 4. Create DataLoaders
    # Pin memory enables faster host-to-device transfer for CUDA
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batch to stabilize BatchNorm stats
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
