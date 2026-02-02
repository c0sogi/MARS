import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.features import FeatureEngineer
from library.utils import seed_everything


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for the Ventilator Pressure Prediction task.
    Wraps pre-processed numpy arrays into PyTorch tensors.
    """

    def __init__(self, x, y, u_out):
        """
        Args:
            x (np.ndarray): Input features of shape (N_breaths, Sequence_Length, N_features).
            y (np.ndarray): Target pressure of shape (N_breaths, Sequence_Length).
            u_out (np.ndarray): Expiratory valve status of shape (N_breaths, Sequence_Length).
        """
        # Convert to float32 tensors immediately to save overhead during training loop
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        # u_out is binary (0/1) but used as a multiplicative mask in loss, so float32 is appropriate
        self.u_out = torch.tensor(u_out, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        """
        Returns a dictionary containing the data for a single breath.
        """
        return {"x": self.x[idx], "y": self.y[idx], "u_out": self.u_out[idx]}


def prepare_datasets(load_cached_data=True, batch_size=None, num_workers=None):
    """
    Orchestrates data loading, feature engineering, and DataLoader creation.

    Args:
        load_cached_data (bool): Whether to try loading pre-processed .npy files from cache.
        batch_size (int, optional): Override Config.BATCH_SIZE.
        num_workers (int, optional): Override Config.NUM_WORKERS.

    Returns:
        tuple: (train_loader, val_loader, test_loader, test_ids)
    """
    # Ensure reproducibility
    seed_everything(Config.SEED)

    # Resolve defaults
    bs = batch_size if batch_size is not None else Config.BATCH_SIZE
    nw = num_workers if num_workers is not None else Config.NUM_WORKERS

    # Initialize Feature Engineer
    fe = FeatureEngineer()

    # --- Load and Process Data ---
    # The FeatureEngineer handles caching, scaling, and reshaping internally.

    # Train Data
    x_train, y_train, u_out_train, _ = fe.get_data(
        "train", load_cached_data=load_cached_data
    )

    # Validation Data
    x_val, y_val, u_out_val, _ = fe.get_data("val", load_cached_data=load_cached_data)

    # Test Data
    x_test, y_test, u_out_test, test_ids = fe.get_data(
        "test", load_cached_data=load_cached_data
    )

    # --- Create Datasets ---
    train_dataset = VentilatorDataset(x_train, y_train, u_out_train)
    val_dataset = VentilatorDataset(x_val, y_val, u_out_val)
    test_dataset = VentilatorDataset(x_test, y_test, u_out_test)

    # --- Create DataLoaders ---
    # Drop last batch in training to maintain consistent batch statistics (e.g., for BatchNorm)
    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, test_ids
