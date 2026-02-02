import torch
from torch.utils.data import DataLoader
from library.model import process_data, GNSSWindowDataset


def get_dataloaders(batch_size=256, window_size=11, load_cached_data=True):
    """
    Constructs DataLoaders for train, validation, and test sets using pre-defined processing logic.

    Args:
        batch_size (int): Number of samples per batch.
        window_size (int): Temporal window size for sequence features.
        load_cached_data (bool): If True, attempts to load pre-processed data from cache.

    Returns:
        tuple: (train_loader, val_loader, test_loader, meta_test)
            - train_loader (DataLoader): Loader for training data (shuffled).
            - val_loader (DataLoader): Loader for validation data.
            - test_loader (DataLoader): Loader for test data.
            - meta_test (pd.DataFrame): Metadata for test set (tripId, UnixTimeMillis, WlsLat, WlsLon).
    """

    # ---------------------------------------------------------
    # 1. Train Loader
    # ---------------------------------------------------------
    # process_data handles caching, scaling, and windowing internally.
    # It fits the scaler on train data and saves stats for val/test.
    X_train, y_train, _ = process_data(
        mode="train", window_size=window_size, load_cached_data=load_cached_data
    )

    train_dataset = GNSSWindowDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # ---------------------------------------------------------
    # 2. Validation Loader
    # ---------------------------------------------------------
    # Loads scaler stats fitted on train data
    X_val, y_val, _ = process_data(
        mode="val", window_size=window_size, load_cached_data=load_cached_data
    )

    val_dataset = GNSSWindowDataset(X_val, y_val)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # ---------------------------------------------------------
    # 3. Test Loader
    # ---------------------------------------------------------
    # Loads scaler stats fitted on train data
    # Returns meta_test which is needed for submission generation (WLS baseline)
    X_test, _, meta_test = process_data(
        mode="test", window_size=window_size, load_cached_data=load_cached_data
    )

    # Test dataset has no targets (y=None)
    test_dataset = GNSSWindowDataset(X_test, y=None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, meta_test
