import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from library.config import Config
from library.utils import preprocess_data


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA Degradation Prediction.
    Wraps preprocessed numpy arrays for features, structure indices, and targets.
    """

    def __init__(self, data_dict, is_test=False):
        """
        Args:
            data_dict (dict): Dictionary containing 'features', 'pair_indices', 'ids',
                              and optionally 'targets'.
            is_test (bool): Flag indicating if this is the test set (no targets).
        """
        self.features = data_dict["features"]
        self.pair_indices = data_dict["pair_indices"]
        self.ids = data_dict["ids"]
        self.is_test = is_test

        if not self.is_test:
            self.targets = data_dict["targets"]
        else:
            self.targets = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """
        Returns a single sample.

        Returns:
            dict: {
                'x': Tensor (Seq_Len, Channels),
                'pair_indices': Tensor (Seq_Len,),
                'id': str,
                'y': Tensor (Seq_Scored, Num_Targets) [Only if not test]
            }
        """
        # Features: (Seq_Len, Channels) -> Float32
        x = torch.tensor(self.features[idx], dtype=torch.float32)

        # Pair Indices: (Seq_Len,) -> Long
        pair_indices = torch.tensor(self.pair_indices[idx], dtype=torch.long)

        sample = {"x": x, "pair_indices": pair_indices, "id": self.ids[idx]}

        if not self.is_test:
            # Targets: (Seq_Scored, Num_Targets) -> Float32
            # Note: Targets are only provided for the first 68 positions
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            sample["y"] = y

        return sample


def get_loaders(load_cached_data=True):
    """
    Creates and returns DataLoaders for training and validation splits.

    Args:
        load_cached_data (bool): Whether to load preprocessed data from cache.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Load data using utility function which handles caching and processing
    train_data = preprocess_data("train", load_cached_data=load_cached_data)
    val_data = preprocess_data("val", load_cached_data=load_cached_data)

    # Handle Debugging: Subset data to speed up development loop
    if Config.DEBUG:
        subset_size = min(Config.DEBUG_SUBSET_SIZE, len(train_data["ids"]))
        for key in train_data:
            train_data[key] = train_data[key][:subset_size]

        subset_size_val = min(Config.DEBUG_SUBSET_SIZE, len(val_data["ids"]))
        for key in val_data:
            val_data[key] = val_data[key][:subset_size_val]

    # Create Datasets
    train_dataset = RNADataset(train_data, is_test=False)
    val_dataset = RNADataset(val_data, is_test=False)

    # Create DataLoaders
    # Pin memory speeds up host-to-device transfer
    use_pin_memory = Config.DEVICE == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=use_pin_memory,
        drop_last=True,  # Drop incomplete batch to maintain consistent batch statistics
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,  # Sequential for validation
        num_workers=Config.NUM_WORKERS,
        pin_memory=use_pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader(load_cached_data=True):
    """
    Creates and returns DataLoader for the test split.

    Args:
        load_cached_data (bool): Whether to load preprocessed data from cache.

    Returns:
        DataLoader: test_loader
    """
    test_data = preprocess_data("test", load_cached_data=load_cached_data)

    # Handle Debugging for test set
    if Config.DEBUG:
        subset_size = min(Config.DEBUG_SUBSET_SIZE, len(test_data["ids"]))
        for key in test_data:
            test_data[key] = test_data[key][:subset_size]

    test_dataset = RNADataset(test_data, is_test=True)

    use_pin_memory = Config.DEVICE == "cuda"

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=use_pin_memory,
        drop_last=False,
    )

    return test_loader
