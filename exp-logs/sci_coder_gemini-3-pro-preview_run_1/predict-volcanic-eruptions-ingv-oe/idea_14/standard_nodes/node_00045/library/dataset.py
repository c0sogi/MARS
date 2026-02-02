import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from library.config import Config
from library.utils import log1p_target


class SeismicDataset(Dataset):
    """
    PyTorch Dataset for loading pre-processed seismic data.

    Loads .npy files containing:
    1. 'image': Dual-resolution spectrograms (20 channels: 10 sensors * 2 resolutions).
    2. 'scalars': Fusion scalar features (30 dimensions: 10 sensors * 3 stats).
    3. 'target': Time to eruption.

    Applies log1p transformation to the target if configured and not in test mode.
    """

    def __init__(self, metadata, data_dir, mode="train"):
        """
        Args:
            metadata (pd.DataFrame): DataFrame containing 'segment_id' and 'time_to_eruption'.
            data_dir (str): Directory path where the .npy files are stored.
            mode (str): One of 'train', 'val', 'test'. Determines target processing.
        """
        self.metadata = metadata.reset_index(drop=True)
        self.data_dir = data_dir
        self.mode = mode
        self.config = Config

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        segment_id = int(row["segment_id"])

        # Construct file path
        file_path = os.path.join(self.data_dir, f"{segment_id}.npy")

        # Load data
        # We use allow_pickle=True because the data is saved as a dictionary
        try:
            data = np.load(file_path, allow_pickle=True).item()
        except FileNotFoundError:
            # Fallback for missing files (should not happen if preprocessing is complete)
            # Return zeros to prevent crashing, but print error
            print(f"Warning: File not found {file_path}")
            image = np.zeros(
                (
                    self.config.IN_CHANNELS,
                    self.config.IMG_SIZE[0],
                    self.config.IMG_SIZE[1],
                ),
                dtype=np.float32,
            )
            scalars = np.zeros((self.config.SCALAR_DIM,), dtype=np.float32)
            target = 0.0
            return (
                torch.tensor(image),
                torch.tensor(scalars),
                torch.tensor(target, dtype=torch.float32),
            )

        # Extract components
        image = data["image"]  # Shape: (20, 128, 128)
        scalars = data["scalars"]  # Shape: (30,)
        target = data["target"]  # Scalar

        # Transform Target
        if self.mode != "test" and self.config.LOG_SCALE_TARGET:
            target = log1p_target(target)

        # Convert to Tensors
        image_tensor = torch.from_numpy(image).float()
        scalar_tensor = torch.from_numpy(scalars).float()
        target_tensor = torch.tensor(target, dtype=torch.float32)

        return image_tensor, scalar_tensor, target_tensor


def get_dataloader(split, batch_size=None, shuffle=None, num_workers=None):
    """
    Factory function to create DataLoaders for specific splits.

    Args:
        split (str): 'train', 'val', or 'test'.
        batch_size (int, optional): Batch size. Defaults to Config.BATCH_SIZE.
        shuffle (bool, optional): Whether to shuffle. Defaults to True for train, False otherwise.
        num_workers (int, optional): Number of workers. Defaults to Config.NUM_WORKERS.

    Returns:
        DataLoader: PyTorch DataLoader instance.
    """
    config = Config

    # Set defaults if not provided
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if num_workers is None:
        num_workers = config.NUM_WORKERS

    # Determine paths and settings based on split
    if split == "train":
        meta_path = config.TRAIN_METADATA_PATH
        data_dir = config.SPECTROGRAM_TRAIN_DIR
        mode = "train"
        if shuffle is None:
            shuffle = True
    elif split == "val":
        meta_path = config.VAL_METADATA_PATH
        data_dir = config.SPECTROGRAM_VAL_DIR
        mode = "val"
        if shuffle is None:
            shuffle = False
    elif split == "test":
        meta_path = config.TEST_METADATA_PATH
        data_dir = config.SPECTROGRAM_TEST_DIR
        mode = "test"
        if shuffle is None:
            shuffle = False
    else:
        raise ValueError(f"Unknown split: {split}")

    # Load Metadata
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df = pd.read_csv(meta_path)

    # Debug Mode: specific to training/val usually, but applied generally here if requested
    if config.DEBUG:
        df = df.head(config.DEBUG_SAMPLE_SIZE)

    # Create Dataset
    dataset = SeismicDataset(metadata=df, data_dir=data_dir, mode=mode)

    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if config.DEVICE == "cuda" else False,
        drop_last=(split == "train"),  # Drop last incomplete batch only during training
    )

    return loader
