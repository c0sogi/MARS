import torch
from torch.utils.data import DataLoader
from library.config import Config
from library.model import GNSSWindowDataset as _GNSSWindowDataset

# Expose the GNSSWindowDataset class from library.model to satisfy the module interface
# without re-implementing the logic.
GNSSWindowDataset = _GNSSWindowDataset


def get_dataloader(
    X,
    y=None,
    batch_size=Config.BATCH_SIZE,
    shuffle=False,
    num_workers=Config.NUM_WORKERS,
    pin_memory=None,
):
    """
    Creates and returns a PyTorch DataLoader for the GNSS dataset.

    Args:
        X (np.ndarray): Input features tensor of shape (N, Window_Size, Channels).
        y (np.ndarray, optional): Target labels tensor of shape (N, Output_Dim). Defaults to None.
        batch_size (int, optional): Number of samples per batch. Defaults to Config.BATCH_SIZE.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to False.
        num_workers (int, optional): Number of subprocesses for data loading. Defaults to Config.NUM_WORKERS.
        pin_memory (bool, optional): Whether to copy tensors into CUDA pinned memory.
                                     If None, determined automatically based on CUDA availability.

    Returns:
        DataLoader: A PyTorch DataLoader configured for the GNSS dataset.
    """
    # Instantiate the dataset using the pre-defined class
    # The class handles conversion of numpy arrays to FloatTensors
    dataset = GNSSWindowDataset(X, y)

    # Determine pin_memory automatically if not specified
    # Pinning memory speeds up transfer to GPU
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    # Create the DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return loader
