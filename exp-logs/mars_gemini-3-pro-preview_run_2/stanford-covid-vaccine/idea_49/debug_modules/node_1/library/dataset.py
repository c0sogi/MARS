import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.data_processor import process_data


class RNADataset(Dataset):
    def __init__(self, mode="train", load_cached_data=True, debug=False):
        """
        Dataset class for RNA degradation prediction.

        Args:
            mode (str): One of 'train', 'val', 'test'.
            load_cached_data (bool): Whether to load preprocessed data from cache.
            debug (bool): If True, restricts dataset to a small subset for debugging.
        """
        self.mode = mode

        # Load data using the centralized processor from library
        # process_data handles caching, metadata loading, and feature engineering
        data_dict = process_data(mode=mode, load_cached_data=load_cached_data)

        self.inputs = data_dict["inputs"]
        self.partner_indices = data_dict["partner_indices"]
        self.ids = data_dict["ids"]
        self.targets = data_dict.get("targets")

        # Apply debug limit if requested
        if debug:
            limit = 128  # Small subset size for debugging
            self.inputs = self.inputs[:limit]
            self.partner_indices = self.partner_indices[:limit]
            self.ids = self.ids[:limit]
            if self.targets is not None:
                self.targets = self.targets[:limit]

        # Convert numpy arrays to PyTorch tensors
        self.inputs = torch.tensor(self.inputs, dtype=torch.float32)
        self.partner_indices = torch.tensor(self.partner_indices, dtype=torch.long)

        if self.targets is not None:
            self.targets = torch.tensor(self.targets, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset.

        Returns:
            dict: Dictionary containing:
                - 'inputs': Tensor of shape (L, 18)
                - 'partner_indices': Tensor of shape (L,)
                - 'targets': Tensor of shape (L, 5) (if available)
        """
        item = {
            "inputs": self.inputs[idx],
            "partner_indices": self.partner_indices[idx],
        }

        if self.targets is not None:
            item["targets"] = self.targets[idx]

        return item


def get_dataloader(
    mode="train",
    load_cached_data=True,
    debug=False,
    batch_size=None,
    num_workers=2,
    shuffle=None,
):
    """
    Creates a DataLoader for the RNADataset.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to use cached data.
        debug (bool): Debug mode flag.
        batch_size (int, optional): Batch size. Defaults to Config.BATCH_SIZE.
        num_workers (int): Number of worker threads.
        shuffle (bool, optional): Whether to shuffle. Defaults to True for train, False otherwise.

    Returns:
        DataLoader: Configured PyTorch DataLoader.
    """
    dataset = RNADataset(mode=mode, load_cached_data=load_cached_data, debug=debug)

    if batch_size is None:
        batch_size = Config.BATCH_SIZE

    if shuffle is None:
        shuffle = mode == "train"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
