import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.data_processor import DataProcessor


class ContactDataset(Dataset):
    """
    PyTorch Dataset for the HPI-RVN architecture.

    It manages four distinct input streams corresponding to the hierarchical
    physics-informed groups and the visual stream:
    1. Geometry (Group A): High-precision positional features.
    2. Motion (Group B): First-order derivatives.
    3. Dynamics (Group C): Clamped high-order derivatives.
    4. Visual: Helmet bounding box metrics.
    """

    def __init__(self, features, labels=None):
        """
        Args:
            features (tuple): A tuple containing four numpy arrays:
                              (X_geometry, X_motion, X_dynamics, X_visual).
            labels (numpy.ndarray, optional): Binary target labels.
        """
        self.X_geometry = features[0]
        self.X_motion = features[1]
        self.X_dynamics = features[2]
        self.X_visual = features[3]
        self.labels = labels

    def __len__(self):
        return len(self.X_geometry)

    def __getitem__(self, idx):
        # Convert numpy arrays to FloatTensors
        sample = {
            "geometry": torch.tensor(self.X_geometry[idx], dtype=torch.float32),
            "motion": torch.tensor(self.X_motion[idx], dtype=torch.float32),
            "dynamics": torch.tensor(self.X_dynamics[idx], dtype=torch.float32),
            "visual": torch.tensor(self.X_visual[idx], dtype=torch.float32),
        }

        if self.labels is not None:
            # BCEWithLogitsLoss expects float targets
            sample["label"] = torch.tensor(self.labels[idx], dtype=torch.float32)

        return sample


def get_dataloader(
    mode="train",
    batch_size=Config.BATCH_SIZE,
    load_cached_data=True,
    shuffle=None,
    num_workers=Config.NUM_WORKERS,
):
    """
    Factory function to create a DataLoader for a specific mode.

    This function utilizes the DataProcessor to load and process data (with caching),
    wraps it in a ContactDataset, and returns a PyTorch DataLoader.

    Args:
        mode (str): 'train', 'validation', or 'test'.
        batch_size (int): Batch size for the loader.
        load_cached_data (bool): Whether to attempt loading from parquet cache.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True for train, False otherwise.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        DataLoader: Configured PyTorch DataLoader.
        pd.DataFrame: The contact_ids corresponding to the dataset (useful for submission).
    """
    # Default shuffle logic
    if shuffle is None:
        shuffle = mode == "train"

    # Initialize Processor
    processor = DataProcessor()

    # Load Data (Delegates to DataProcessor for caching and engineering)
    # Returns: (tuple of arrays), labels, ids_dataframe
    features, y, ids = processor.get_data(mode=mode, load_cached_data=load_cached_data)

    # Create Dataset
    dataset = ContactDataset(features, labels=y)

    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=(num_workers > 0),
    )

    return loader, ids
