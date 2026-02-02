import torch
from torch.utils.data import Dataset, DataLoader
from library.data_processing import DataProcessor
from library.config import BATCH_SIZE, NUM_WORKERS


class ContactDataset(Dataset):
    """
    PyTorch Dataset for the Contact Detection task.
    Wraps pre-processed tensors for kinematic, visual, and categorical streams provided by DataProcessor.
    """

    def __init__(self, data_dict):
        """
        Args:
            data_dict (dict): Dictionary containing:
                - 'X_kin': Tensor of shape (N, F_kin)
                - 'X_vis': Tensor of shape (N, F_vis)
                - 'X_cat': Tensor of shape (N, F_cat)
                - 'y': Tensor of shape (N,)
                - 'ids': Array of contact_ids (optional, for inference)
        """
        self.X_kin = data_dict["X_kin"]
        self.X_vis = data_dict["X_vis"]
        self.X_cat = data_dict["X_cat"]
        self.y = data_dict["y"]

        # Handle IDs: If present, keep them. If not, create placeholder.
        # DataLoader handles lists of strings correctly in default_collate.
        self.ids = data_dict.get("ids")
        if self.ids is None:
            # Create a placeholder list of empty strings to maintain consistency
            self.ids = ["" for _ in range(len(self.y))]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "X_kin": self.X_kin[idx],
            "X_vis": self.X_vis[idx],
            "X_cat": self.X_cat[idx],
            "y": self.y[idx],
            "contact_id": self.ids[idx],
        }


def get_dataloaders(
    debug=False, load_cached_data=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
):
    """
    Initializes the DataProcessor, loads train/val data, and returns DataLoaders.

    Args:
        debug (bool): If True, uses a smaller subset of data.
        load_cached_data (bool): If True, attempts to load from parquet cache.
        batch_size (int): Batch size for the dataloaders.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        tuple: (train_loader, val_loader)
    """
    processor = DataProcessor(debug=debug)

    # Load processed data (returns dict of tensors)
    # The DataProcessor handles the caching logic internally based on the flag
    train_data = processor.get_data(split="train", load_cached_data=load_cached_data)
    val_data = processor.get_data(split="val", load_cached_data=load_cached_data)

    # Create Datasets
    train_dataset = ContactDataset(train_data)
    val_dataset = ContactDataset(val_data)

    # Create DataLoaders
    # Pin memory improves transfer speed to GPU
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    return train_loader, val_loader


def get_test_loader(
    debug=False, load_cached_data=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
):
    """
    Initializes the DataProcessor, loads test data, and returns the Test DataLoader.

    Args:
        debug (bool): If True, uses a smaller subset of data.
        load_cached_data (bool): If True, attempts to load from parquet cache.
        batch_size (int): Batch size for the dataloader.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        DataLoader: The test data loader.
    """
    processor = DataProcessor(debug=debug)

    # Load processed test data
    test_data = processor.get_data(split="test", load_cached_data=load_cached_data)

    # Create Dataset
    test_dataset = ContactDataset(test_data)

    # Create DataLoader
    use_pin_memory = torch.cuda.is_available()

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    return test_loader
