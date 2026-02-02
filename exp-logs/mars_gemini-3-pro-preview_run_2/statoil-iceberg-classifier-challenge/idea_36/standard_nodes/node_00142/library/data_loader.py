import json
import pandas as pd
import torch
from torch.utils.data import DataLoader
from library.config import (
    TRAIN_JSON,
    TRAIN_META_PATH,
    VAL_META_PATH,
    SEED,
    NUM_WORKERS,
    set_seed,
)
from library.model import load_and_process_data, IcebergDataset


def get_dataloaders(batch_size=32, load_cached_data=True):
    """
    Creates DataLoaders for Train, Validation, and Test sets.
    Uses the fixed splits defined in ./metadata/train.csv and ./metadata/val.csv.

    Args:
        batch_size (int): Batch size for the dataloaders.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    set_seed(SEED)

    # 1. Load processed (and normalized) data
    # X_train_full contains the entire training set (train + val from original source)
    # The load_and_process_data function handles caching and global normalization internally.
    X_train_full, y_train_full, inc_train_full, X_test, inc_test, test_ids = (
        load_and_process_data(load_cached_data)
    )

    # 2. Map IDs to indices in the full training arrays
    # We need to read train.json again to know the order of X_train_full to map metadata IDs to indices
    with open(TRAIN_JSON, "r") as f:
        train_json_raw = json.load(f)

    # Create a mapping from ID to index
    id_to_idx = {item["id"]: i for i, item in enumerate(train_json_raw)}

    # 3. Load Metadata Splits
    df_train_meta = pd.read_csv(TRAIN_META_PATH)
    df_val_meta = pd.read_csv(VAL_META_PATH)

    # 4. Filter Indices based on Metadata
    # Get the indices in the full arrays that correspond to the train and val splits
    train_indices = [id_to_idx[uid] for uid in df_train_meta["id"] if uid in id_to_idx]
    val_indices = [id_to_idx[uid] for uid in df_val_meta["id"] if uid in id_to_idx]

    # 5. Subset the Arrays
    X_train = X_train_full[train_indices]
    y_train = y_train_full[train_indices]
    inc_train = inc_train_full[train_indices]

    X_val = X_train_full[val_indices]
    y_val = y_train_full[val_indices]
    inc_val = inc_train_full[val_indices]

    # 6. Create Datasets
    # Train: Transform = True (Applies Random Rotation/Flip)
    train_ds = IcebergDataset(X_train, y_train, inc_train, transform=True)
    # Val: Transform = False
    val_ds = IcebergDataset(X_val, y_val, inc_val, transform=False)
    # Test: Transform = False
    test_ds = IcebergDataset(X_test, None, inc_test, transform=False)

    # 7. Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader
