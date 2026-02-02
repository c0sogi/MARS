import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from library.config import Config
from library.model import get_data


def get_dataloaders(
    load_cached=True,
    batch_size=Config.BATCH_SIZE,
    debug_samples=Config.MAX_DEBUG_SAMPLES,
):
    """
    Orchestrates the loading, processing, and batching of data for the Parallel DCN-ResNet.

    This function leverages the centralized get_data utility to perform Physics-Informed
    Feature Engineering and caching, then prepares PyTorch DataLoaders for the training loop.

    Args:
        load_cached (bool): If True, attempts to load pre-processed data from the cache directory.
                            If False or cache is missing, processes data from scratch.
        batch_size (int): The number of samples per batch.
        debug_samples (int or None): If set, limits the dataset size for rapid debugging.

    Returns:
        train_loader (DataLoader): Shuffled loader for the training set.
        val_loader (DataLoader): Sequential loader for the validation set.
        test_loader (DataLoader): Sequential loader for the test set.
        test_ids (np.ndarray): The original IDs for the test set rows (required for submission).
        class_map (dict): A dictionary mapping original class labels to 0-indexed targets.
    """

    # 1. Retrieve Processed Data
    # Delegates loading, feature engineering, scaling, and caching to the library model.
    X_train, y_train_raw, X_val, y_val_raw, X_test, test_ids = get_data(
        load_cached_data=load_cached, debug_samples=debug_samples
    )

    # 2. Target Encoding
    # Map potentially non-contiguous class labels (e.g., 1, 2, 3, 7) to 0-indexed integers (0, 1, 2, 3).
    # We derive the map from both train and val sets to ensure all classes are covered.
    unique_classes = sorted(np.unique(np.concatenate([y_train_raw, y_val_raw])))
    class_map = {c: i for i, c in enumerate(unique_classes)}

    y_train = np.array([class_map[c] for c in y_train_raw], dtype=np.int64)
    y_val = np.array([class_map[c] for c in y_val_raw], dtype=np.int64)

    # 3. Tensor Creation
    # Features are already cast to float32 in get_data.
    train_x_tensor = torch.from_numpy(X_train)
    train_y_tensor = torch.from_numpy(y_train)

    val_x_tensor = torch.from_numpy(X_val)
    val_y_tensor = torch.from_numpy(y_val)

    test_x_tensor = torch.from_numpy(X_test)

    # 4. Dataset Wrapping
    train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
    val_dataset = TensorDataset(val_x_tensor, val_y_tensor)
    test_dataset = TensorDataset(test_x_tensor)

    # 5. DataLoader Construction
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, test_ids, class_map
