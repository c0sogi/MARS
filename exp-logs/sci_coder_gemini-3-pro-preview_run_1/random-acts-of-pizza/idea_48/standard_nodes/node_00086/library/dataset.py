import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, Tuple
from library.config import Config


class PizzaDataset(Dataset):
    def __init__(self, data_dict: Dict[str, Any], is_test: bool = False):
        """
        PyTorch Dataset for the Pizza Request task.

        Args:
            data_dict (Dict[str, Any]): Dictionary containing feature arrays.
                Expected keys: 'title_emb', 'body_emb', 'history_seq',
                'history_mask', 'history_cent', 'metadata', and optionally 'y'.
            is_test (bool): If True, 'y' (target) is not expected/loaded.
        """
        self.is_test = is_test

        # Convert numpy arrays to tensors
        # We use float32 for features to match default torch float type
        self.title_emb = torch.tensor(data_dict["title_emb"], dtype=torch.float32)
        self.body_emb = torch.tensor(data_dict["body_emb"], dtype=torch.float32)
        self.history_seq = torch.tensor(data_dict["history_seq"], dtype=torch.float32)
        self.history_mask = torch.tensor(data_dict["history_mask"], dtype=torch.float32)
        self.history_cent = torch.tensor(data_dict["history_cent"], dtype=torch.float32)
        self.metadata = torch.tensor(data_dict["metadata"], dtype=torch.float32)

        if not self.is_test:
            # Target should be float for BCEWithLogitsLoss
            # Input y is typically int array (0/1), convert to float
            self.y = torch.tensor(data_dict["y"], dtype=torch.float32)
        else:
            self.y = None

        self.n_samples = self.title_emb.shape[0]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = {
            "title_emb": self.title_emb[idx],
            "body_emb": self.body_emb[idx],
            "history_seq": self.history_seq[idx],
            "history_mask": self.history_mask[idx],
            "history_cent": self.history_cent[idx],
            "metadata": self.metadata[idx],
        }

        if not self.is_test:
            sample["target"] = self.y[idx]

        return sample


def get_dataloaders(
    mlp_data: Dict[str, Any], batch_size: int = Config.BATCH_SIZE, num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        mlp_data (Dict[str, Any]): Dictionary containing 'train', 'val', 'test' data dictionaries.
            Handles cases where inner dicts are wrapped in 0-d numpy arrays (from np.savez).
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.

    Returns:
        train_loader, val_loader, test_loader
    """

    def extract_data(key):
        if key not in mlp_data:
            raise KeyError(f"Key '{key}' not found in mlp_data.")
        val = mlp_data[key]
        # If it's a 0-d numpy array (saved object), extract item
        if isinstance(val, np.ndarray) and val.ndim == 0:
            return val.item()
        return val

    train_data = extract_data("train")
    val_data = extract_data("val")
    test_data = extract_data("test")

    train_dataset = PizzaDataset(train_data, is_test=False)
    val_dataset = PizzaDataset(val_data, is_test=False)
    test_dataset = PizzaDataset(test_data, is_test=True)

    # Pin memory improves transfer speed to GPU
    pin_memory = torch.cuda.is_available()

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

    return train_loader, val_loader, test_loader
