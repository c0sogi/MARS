import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional

from library.config import Config
from library.features import prepare_train_data, prepare_val_data, prepare_test_data


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for the Ventilator Pressure Prediction task.
    Wraps preprocessed numpy arrays and provides tensors for the model.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        u_out: Optional[np.ndarray] = None,
        ids: Optional[np.ndarray] = None,
        is_test: bool = False,
    ):
        """
        Args:
            X (np.ndarray): Feature tensor of shape (N_breaths, 80, N_features).
            y (np.ndarray, optional): Target pressure tensor of shape (N_breaths, 80).
            u_out (np.ndarray, optional): Expiratory mask tensor of shape (N_breaths, 80).
            ids (np.ndarray, optional): ID tensor of shape (N_breaths, 80).
            is_test (bool): Flag indicating if this is a test dataset (returns IDs instead of targets).
        """
        # Convert to FloatTensor for model input
        self.X = torch.tensor(X, dtype=torch.float32)
        self.is_test = is_test

        # Targets (Pressure)
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = None

        # Expiratory Phase Mask (u_out)
        if u_out is not None:
            self.u_out = torch.tensor(u_out, dtype=torch.float32)
        else:
            self.u_out = None

        # Time Step IDs
        if ids is not None:
            # IDs are needed as Long/Int for submission mapping
            self.ids = torch.tensor(ids, dtype=torch.long)
        else:
            self.ids = None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        """
        Returns a tuple of tensors.
        Train/Val: (X, y, u_out)
        Test:      (X, u_out, ids)
        """
        if self.is_test:
            # For inference: Features, Mask (for consistency/masking), IDs (for submission)
            return self.X[idx], self.u_out[idx], self.ids[idx]
        else:
            # For training: Features, Target, Mask (for loss calculation)
            return self.X[idx], self.y[idx], self.u_out[idx]


def get_data_loaders(
    batch_size: int = Config.BATCH_SIZE,
    num_workers: int = Config.NUM_WORKERS,
    load_cached_data: bool = True,
    debug_limit: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Orchestrates data loading, preprocessing, and DataLoader creation.
    Delegates heavy lifting to library.features.prepare_* functions which handle caching.

    Args:
        batch_size (int): Batch size for training and inference.
        num_workers (int): Number of subprocesses for data loading.
        load_cached_data (bool): Whether to use cached .npy files if available.
        debug_limit (int, optional): Limit the number of samples for debugging purposes.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, test_loader)
    """
    # 1. Prepare Training Data
    # This loads raw data, computes features, scales inputs, and saves cache.
    # It also fits and returns the scaler used for normalization.
    train_X, train_y, train_uout, train_ids, scaler = prepare_train_data(
        load_cached_data=load_cached_data
    )

    # 2. Prepare Validation Data
    # Uses the scaler fitted on training data.
    val_X, val_y, val_uout, val_ids = prepare_val_data(
        scaler=scaler, load_cached_data=load_cached_data
    )

    # 3. Prepare Test Data
    # Uses the scaler fitted on training data.
    test_X, test_uout, test_ids = prepare_test_data(
        scaler=scaler, load_cached_data=load_cached_data
    )

    # 4. Apply Debug Limit (if requested)
    if debug_limit is not None:
        train_X, train_y, train_uout, train_ids = (
            train_X[:debug_limit],
            train_y[:debug_limit],
            train_uout[:debug_limit],
            train_ids[:debug_limit],
        )
        val_X, val_y, val_uout, val_ids = (
            val_X[:debug_limit],
            val_y[:debug_limit],
            val_uout[:debug_limit],
            val_ids[:debug_limit],
        )
        test_X, test_uout, test_ids = (
            test_X[:debug_limit],
            test_uout[:debug_limit],
            test_ids[:debug_limit],
        )

    # 5. Create Dataset Instances
    train_dataset = VentilatorDataset(
        train_X, train_y, train_uout, train_ids, is_test=False
    )
    val_dataset = VentilatorDataset(val_X, val_y, val_uout, val_ids, is_test=False)
    test_dataset = VentilatorDataset(test_X, None, test_uout, test_ids, is_test=True)

    # 6. Create DataLoaders
    # Use pin_memory=True for faster host-to-device transfer on CUDA
    pin_memory = Config.PIN_MEMORY and torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop incomplete batch to maintain consistent stats
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
