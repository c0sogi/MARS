import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from library.config import Config
from library.utils import load_dataset


class SiameseMRIDataset(Dataset):
    """
    PyTorch Dataset for the Siamese Spatially-Strided 2.5D Network.

    Expects input data X of shape (N, 2, C, H, W), where:
    - Dim 1 (size 2) represents the Even (0) and Odd (1) streams.
    - C is the channel depth (64: 16 slices * 4 modalities).
    """

    def __init__(self, X, y=None, ids=None):
        self.X = X
        self.y = y
        self.ids = ids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # X[idx] has shape (2, 64, 224, 224)
        # 0 -> Even Stream, 1 -> Odd Stream
        even_stream = self.X[idx, 0]
        odd_stream = self.X[idx, 1]

        # Convert to FloatTensor
        even_tensor = torch.from_numpy(even_stream).float()
        odd_tensor = torch.from_numpy(odd_stream).float()

        sample = {
            "even": even_tensor,
            "odd": odd_tensor,
            "BraTS21ID": self.ids[idx] if self.ids is not None else "",
        }

        if self.y is not None:
            # Target needs to be float for BCEWithLogitsLoss
            target = torch.tensor(self.y[idx], dtype=torch.float32)
            sample["target"] = target

        return sample


def get_dataloader(subset, batch_size=None, shuffle=None, load_cached_data=True):
    """
    Factory function to create a DataLoader for a specific subset.

    Args:
        subset (str): 'train', 'val', or 'test'.
        batch_size (int, optional): Overrides Config.BATCH_SIZE if provided.
        shuffle (bool, optional): Overrides default shuffling logic if provided.
        load_cached_data (bool): Whether to use cached .npy files.

    Returns:
        DataLoader: Configured PyTorch DataLoader.
    """
    # Load data using the utility function which handles caching and processing
    X, y, ids = load_dataset(subset=subset, load_cached_data=load_cached_data)

    # Initialize Dataset
    dataset = SiameseMRIDataset(X, y, ids)

    # Determine DataLoader parameters
    if batch_size is None:
        batch_size = Config.BATCH_SIZE

    if shuffle is None:
        # Default: Shuffle train, don't shuffle val/test
        shuffle = subset == "train"

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    return loader
