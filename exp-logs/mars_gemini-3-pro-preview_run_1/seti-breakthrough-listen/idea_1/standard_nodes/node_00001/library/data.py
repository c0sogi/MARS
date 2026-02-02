import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class ETDataset(Dataset):
    """
    Custom Dataset for SETI Technosignature Detection.
    Implements the Spatial Difference preprocessing strategy:
    1. Aggregates ON and OFF panels.
    2. Computes Difference Map (ON - OFF).
    3. Normalizes the result.
    """

    def __init__(self, metadata_path, mode="train", debug=False):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            mode (str): One of 'train', 'val', 'test'.
            debug (bool): If True, use a small subset of the data.
        """
        self.mode = mode
        self.input_dir = Config.INPUT_DIR

        # Load metadata
        try:
            self.metadata = pd.read_csv(metadata_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

        # Handle Debug Mode
        if debug:
            subset_size = 100  # Small subset for debugging
            if len(self.metadata) > subset_size:
                self.metadata = self.metadata.sample(
                    n=subset_size, random_state=Config.SEED
                ).reset_index(drop=True)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        file_rel_path = row["file_path"]
        file_path = os.path.join(self.input_dir, file_rel_path)

        # 1. Load Data
        # Data is float16, convert to float32 for processing
        try:
            raw_data = np.load(file_path).astype(np.float32)
        except Exception as e:
            # Fallback for missing files (should not happen based on metadata check)
            # Return zeros matching RAW_SHAPE
            raw_data = np.zeros(Config.RAW_SHAPE, dtype=np.float32)

        # 2. Preprocessing: Spatial Difference
        # Raw shape: (6, 273, 256)
        # ON panels: 0, 2, 4
        # OFF panels: 1, 3, 5

        on_panels = raw_data[[0, 2, 4], :, :]
        off_panels = raw_data[[1, 3, 5], :, :]

        mean_on = np.mean(on_panels, axis=0)
        mean_off = np.mean(off_panels, axis=0)

        # Difference Map
        diff_map = mean_on - mean_off

        # 3. Normalization (Instance Level)
        # Standardize to zero mean and unit variance
        mean_val = np.mean(diff_map)
        std_val = np.std(diff_map)

        # Avoid division by zero
        if std_val < 1e-8:
            std_val = 1.0

        normalized_map = (diff_map - mean_val) / std_val

        # Add channel dimension: (1, 273, 256)
        image = normalized_map[np.newaxis, :, :]

        # Convert to Tensor
        image_tensor = torch.from_numpy(image)

        # Return based on mode
        if self.mode in ["train", "val"]:
            target = row["target"]
            # Return target as float for BCEWithLogitsLoss
            return image_tensor, torch.tensor(target, dtype=torch.float32)
        else:
            # For test, return image and ID
            sample_id = row["id"]
            return image_tensor, sample_id


def get_dataloaders(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, debug=Config.DEBUG
):
    """
    Factory function to create DataLoaders for train, val, and test sets.

    Args:
        batch_size (int): Batch size for loading.
        num_workers (int): Number of subprocesses for data loading.
        debug (bool): Whether to run in debug mode (subset of data).

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """

    # Initialize Datasets
    train_dataset = ETDataset(
        metadata_path=Config.TRAIN_METADATA_PATH, mode="train", debug=debug
    )

    val_dataset = ETDataset(
        metadata_path=Config.VAL_METADATA_PATH, mode="val", debug=debug
    )

    test_dataset = ETDataset(
        metadata_path=Config.TEST_METADATA_PATH, mode="test", debug=debug
    )

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True,  # Drop incomplete batches during training
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader
