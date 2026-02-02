import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.data_processing import prepare_data


class ContactDataset(Dataset):
    """
    PyTorch Dataset wrapper for the NFL Contact Detection task.

    Handles the specific input format required by the CK-ResNet model:
    - Wide temporal features (t-5 to t+5)
    - Center frame features (t=0)
    - Conditioning vector (is_ground)
    """

    def __init__(self, inputs, targets):
        """
        Args:
            inputs (tuple): A tuple containing (X_wide, X_center, condition) tensors.
            targets (torch.Tensor or np.array): Target labels (for train/val) or contact_ids (for test).
        """
        self.x_wide = inputs[0]
        self.x_center = inputs[1]
        self.condition = inputs[2]
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        """
        Returns:
            tuple: ((x_wide, x_center, condition), target)
        """
        # Retrieve features for the specific index
        x_w = self.x_wide[idx]
        x_c = self.x_center[idx]
        cond = self.condition[idx]

        target = self.targets[idx]

        return (x_w, x_c, cond), target


def get_dataloaders(
    load_cached_data=True,
    debug=False,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
):
    """
    Orchestrates data loading, processing, and DataLoader creation.

    Args:
        load_cached_data (bool): Whether to load pre-processed features from disk.
        debug (bool): If True, uses a small subset of data for debugging.
        batch_size (int): Batch size for the DataLoaders.
        num_workers (int): Number of worker processes for data loading.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 1. Retrieve processed tensors from the data processing library
    # prepare_data handles caching and raw data loading internally
    train_data, val_data, test_data = prepare_data(
        load_cached_data=load_cached_data, debug=debug
    )

    # 2. Instantiate Datasets
    # train_data is ((X_wide, X_center, cond), targets)
    train_dataset = ContactDataset(train_data[0], train_data[1])
    val_dataset = ContactDataset(val_data[0], val_data[1])
    test_dataset = ContactDataset(test_data[0], test_data[1])

    # 3. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
