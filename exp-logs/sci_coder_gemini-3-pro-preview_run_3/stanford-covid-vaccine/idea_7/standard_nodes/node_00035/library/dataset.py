import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.features import load_dataset


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA degradation prediction.
    Wraps preprocessed features, targets, and weights.
    """

    def __init__(self, data_dict, split):
        """
        Args:
            data_dict (dict): Dictionary containing data arrays ('X', 'ids', and optionally 'y', 'w').
            split (str): The data split ('train', 'val', or 'test').
        """
        self.split = split
        self.X = data_dict["X"]
        self.ids = data_dict["ids"]

        # Targets and weights are only available for training and validation splits
        if self.split != "test":
            self.y = data_dict["y"]
            self.w = data_dict["w"]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """
        Returns a tuple of data items for the given index.

        Returns:
            train/val: (features, targets, weights)
            test: (features, sample_id)
        """
        # Convert features to tensor. Shape: (107, 28)
        X_sample = torch.tensor(self.X[idx], dtype=torch.float32)

        if self.split == "test":
            # For the test set, we need the ID to map predictions to the submission format
            id_sample = self.ids[idx]
            return X_sample, id_sample
        else:
            # For train/val, return inputs, targets, and error-based weights
            # y Shape: (68, 5)
            # w Shape: (68, 5)
            y_sample = torch.tensor(self.y[idx], dtype=torch.float32)
            w_sample = torch.tensor(self.w[idx], dtype=torch.float32)
            return X_sample, y_sample, w_sample


def get_dataloaders(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, load_cached_data=True
):
    """
    Initializes and returns DataLoaders for train, validation, and test sets.

    Args:
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses for data loading.
        load_cached_data (bool): If True, attempts to load preprocessed data from cache.
                                 If False or cache missing, recomputes from metadata.

    Returns:
        dict: Dictionary containing 'train', 'val', and 'test' DataLoaders.
    """
    # Load data using the library function which handles caching and feature engineering
    train_data = load_dataset("train", load_cached_data=load_cached_data)
    val_data = load_dataset("val", load_cached_data=load_cached_data)
    test_data = load_dataset("test", load_cached_data=load_cached_data)

    # Create Dataset instances
    train_dataset = RNADataset(train_data, split="train")
    val_dataset = RNADataset(val_data, split="val")
    test_dataset = RNADataset(test_data, split="test")

    # Determine if pinned memory should be used (for GPU acceleration)
    pin_memory = torch.cuda.is_available()

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}
