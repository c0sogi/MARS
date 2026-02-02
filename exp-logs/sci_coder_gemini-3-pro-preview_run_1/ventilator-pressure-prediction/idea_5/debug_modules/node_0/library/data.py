import torch
from torch.utils.data import Dataset, DataLoader
from library import config, features


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for the Ventilator Pressure Prediction task.

    Holds preprocessed tensors for:
    - Continuous features (CNN/LSTM branch)
    - Categorical indices (Embeddings)
    - Physics features (Linear Adapter branch)
    - u_out (Masking)
    - ids (Submission mapping)
    - Targets (Pressure)
    """

    def __init__(self, x_cont, x_cat, x_phys, u_out, ids, y=None):
        self.x_cont = torch.tensor(x_cont, dtype=torch.float32)
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        self.x_phys = torch.tensor(x_phys, dtype=torch.float32)
        # u_out is used for masking the loss (we only care about inspiratory phase where u_out=0)
        self.u_out = torch.tensor(u_out, dtype=torch.float32)
        self.ids = torch.tensor(ids, dtype=torch.long)

        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = None

    def __len__(self):
        return len(self.x_cont)

    def __getitem__(self, idx):
        item = {
            "x_cont": self.x_cont[idx],
            "x_cat": self.x_cat[idx],
            "x_phys": self.x_phys[idx],
            "u_out": self.u_out[idx],
            "ids": self.ids[idx],
        }
        if self.y is not None:
            item["y"] = self.y[idx]
        return item


def get_data_loaders(load_cached_data=True, batch_size=config.BATCH_SIZE):
    """
    Loads data using the features library (which handles caching and preprocessing),
    wraps them in VentilatorDataset, and returns PyTorch DataLoaders.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.
        batch_size (int): Batch size for the loaders.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Retrieve dictionary of numpy arrays from the features module
    # This handles the check for existing .npy files and the hash generation
    data = features.prepare_datasets(load_cached_data=load_cached_data)

    # Initialize Datasets
    train_dataset = VentilatorDataset(
        x_cont=data["train_x_cont"],
        x_cat=data["train_x_cat"],
        x_phys=data["train_x_phys"],
        u_out=data["train_uout"],
        ids=data["train_ids"],
        y=data["train_y"],
    )

    val_dataset = VentilatorDataset(
        x_cont=data["val_x_cont"],
        x_cat=data["val_x_cat"],
        x_phys=data["val_x_phys"],
        u_out=data["val_uout"],
        ids=data["val_ids"],
        y=data["val_y"],
    )

    test_dataset = VentilatorDataset(
        x_cont=data["test_x_cont"],
        x_cat=data["test_x_cat"],
        x_phys=data["test_x_phys"],
        u_out=data["test_uout"],
        ids=data["test_ids"],
        y=None,
    )

    # Initialize DataLoaders
    # Train loader: Shuffle is True, drop_last is True to maintain consistent batch shapes
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # Val/Test loaders: Shuffle is False, drop_last is False to process all data
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
