import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import load_dataset


class BraTSDataset(Dataset):
    """
    PyTorch Dataset for BraTS21 Glioblastoma classification.

    This dataset loads pre-processed MRI volumes. It utilizes the caching mechanism
    implemented in library.utils.load_dataset to avoid redundant and expensive
    DICOM processing.
    """

    def __init__(self, subset, load_cached_data=True):
        """
        Args:
            subset (str): One of 'train', 'val', 'test'.
            load_cached_data (bool): If True, attempts to load pre-processed data from cache.
                                     If False or cache miss, processes data from scratch.
        """
        self.subset = subset

        # Load the dataset using the provided utility function.
        # This function returns numpy arrays:
        # X: (N, 4, 32, 256, 256) - Input volumes
        # y: (N,) or None - Target labels
        # ids: (N,) - BraTS21 IDs
        self.X, self.y, self.ids = load_dataset(
            subset, load_cached_data=load_cached_data
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset.

        Returns:
            tuple:
                - If train/val: (image_tensor, label_tensor)
                - If test: (image_tensor, braTS21ID)
        """
        # Convert numpy array to torch tensor
        # Input shape: (4, 32, 256, 256)
        img = torch.from_numpy(self.X[idx])

        if self.subset == "test":
            # For the test set, we need the ID to map predictions to the submission file
            return img, self.ids[idx]
        else:
            # For train/val, we return the label
            # BCEWithLogitsLoss expects float targets
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            return img, label


def get_dataloader(
    subset, batch_size=None, shuffle=None, num_workers=None, load_cached_data=True
):
    """
    Creates a DataLoader for the specified subset.

    Args:
        subset (str): 'train', 'val', or 'test'.
        batch_size (int, optional): Batch size. Defaults to Config.BATCH_SIZE.
        shuffle (bool, optional): Whether to shuffle data. Defaults to True for train, False otherwise.
        num_workers (int, optional): Number of worker processes. Defaults to Config.NUM_WORKERS.
        load_cached_data (bool, optional): Whether to use cached data. Defaults to True.

    Returns:
        DataLoader: Configured PyTorch DataLoader.
    """
    # Set defaults based on Config if not provided
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    if num_workers is None:
        num_workers = Config.NUM_WORKERS

    # Default shuffle logic: Shuffle training data, sequential for val/test
    if shuffle is None:
        shuffle = subset == "train"

    # Initialize dataset
    dataset = BraTSDataset(subset, load_cached_data=load_cached_data)

    # Create DataLoader
    # pin_memory=True speeds up transfer to GPU
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    return loader
