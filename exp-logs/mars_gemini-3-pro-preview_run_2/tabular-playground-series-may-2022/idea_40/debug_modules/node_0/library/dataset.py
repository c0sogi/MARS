import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config


class ManufacturingDataset(Dataset):
    """
    PyTorch Dataset for the Manufacturing Control task.
    Holds references to the full preprocessed arrays to minimize memory usage.
    """

    def __init__(self, indices, X_cat_all, X_cont_all, y_all, is_test=False):
        self.indices = indices
        self.X_cat_all = X_cat_all
        self.X_cont_all = X_cont_all
        self.y_all = y_all
        self.is_test = is_test

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Map the dataset index to the global array index
        global_idx = self.indices[idx]

        # Retrieve data
        cat_data = self.X_cat_all[global_idx]
        cont_data = self.X_cont_all[global_idx]

        # Convert to tensors
        # Categorical: LongTensor for Embeddings
        # Continuous: FloatTensor for Linear layers
        cat_tensor = torch.tensor(cat_data, dtype=torch.long)
        cont_tensor = torch.tensor(cont_data, dtype=torch.float32)

        if self.is_test:
            # For test set, we might not have valid targets, return dummy or just features
            # Returning 0.0 as dummy target to keep signature consistent
            return cat_tensor, cont_tensor, torch.tensor(0.0, dtype=torch.float32)

        target = self.y_all[global_idx]
        target_tensor = torch.tensor(target, dtype=torch.float32)

        return cat_tensor, cont_tensor, target_tensor


def process_and_cache_data(load_cached_data=True):
    """
    Loads raw data, performs preprocessing (tokenization, normalization),
    and caches the result to disk.
    """
    cache_path = Config.PROCESSED_DATA_PATH

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            data = np.load(cache_path, allow_pickle=True)
            return (
                data["ids"],
                data["X_cat"],
                data["X_cont"],
                data["y"],
                data["id_to_idx"].item(),
            )
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from scratch
    print("Processing data from scratch...")

    # Load raw data
    train_df = pd.read_csv(Config.TRAIN_DATA_PATH)
    test_df = pd.read_csv(Config.TEST_DATA_PATH)

    # Identify columns
    # Continuous features are f_00 to f_30, excluding f_27
    cont_cols = [f"f_{i:02d}" for i in range(31) if i != 27]
    cat_col = "f_27"

    # --- Continuous Feature Normalization ---
    print("Normalizing continuous features...")
    scaler = StandardScaler()

    # Fit on Train, Transform Train and Test
    X_cont_train = scaler.fit_transform(train_df[cont_cols].values).astype(np.float32)
    X_cont_test = scaler.transform(test_df[cont_cols].values).astype(np.float32)

    # --- Categorical Feature Tokenization ---
    print("Tokenizing categorical feature f_27...")

    def tokenize_f27(series):
        # Map 'A'->1, 'B'->2, ... 'Z'->26. 0 is reserved for padding/unknown.
        # Convert series to list of strings, then to list of lists of ordinals
        # This is reasonably fast for 1M rows
        return np.array(
            [[ord(c) - ord("A") + 1 for c in s] for s in series], dtype=np.int32
        )

    X_cat_train = tokenize_f27(train_df[cat_col])
    X_cat_test = tokenize_f27(test_df[cat_col])

    # --- Targets ---
    y_train = train_df["target"].values.astype(np.float32)
    # Test targets are placeholders (NaN or -1)
    y_test = np.full(len(test_df), -1.0, dtype=np.float32)

    # --- IDs ---
    ids_train = train_df["id"].values
    ids_test = test_df["id"].values

    # --- Combine All Data ---
    # We concatenate train and test to create a unified lookup
    all_ids = np.concatenate([ids_train, ids_test])
    all_X_cat = np.concatenate([X_cat_train, X_cat_test])
    all_X_cont = np.concatenate([X_cont_train, X_cont_test])
    all_y = np.concatenate([y_train, y_test])

    # Create ID to Index Map
    # This allows O(1) lookup from metadata ID to array index
    id_to_idx = {id_: idx for idx, id_ in enumerate(all_ids)}

    # --- Save to Cache ---
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        ids=all_ids,
        X_cat=all_X_cat,
        X_cont=all_X_cont,
        y=all_y,
        id_to_idx=id_to_idx,
    )
    print(f"Data processed and saved to {cache_path}")

    return all_ids, all_X_cat, all_X_cont, all_y, id_to_idx


def get_dataloaders(batch_size=Config.BATCH_SIZE, load_cached_data=True):
    """
    Creates DataLoaders for Train, Validation, and Test sets based on metadata splits.

    Args:
        batch_size (int): Batch size for dataloaders.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        train_loader, val_loader, test_loader
    """
    # 1. Get Unified Data Arrays
    ids, X_cat, X_cont, y, id_to_idx = process_and_cache_data(load_cached_data)

    # 2. Load Metadata to get splits
    print("Loading metadata for splits...")
    train_meta = pd.read_csv(Config.TRAIN_META_PATH)
    val_meta = pd.read_csv(Config.VAL_META_PATH)
    test_meta = pd.read_csv(Config.TEST_META_PATH)

    # 3. Map Metadata IDs to Global Indices
    def get_indices(meta_df):
        meta_ids = meta_df["id"].values
        indices = [id_to_idx[mid] for mid in meta_ids if mid in id_to_idx]
        if len(indices) != len(meta_ids):
            print(
                f"Warning: Some IDs in metadata not found in raw data! Found {len(indices)}/{len(meta_ids)}"
            )
        return np.array(indices, dtype=np.int32)

    train_indices = get_indices(train_meta)
    val_indices = get_indices(val_meta)
    test_indices = get_indices(test_meta)

    print(
        f"Split sizes - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}"
    )

    # 4. Create Datasets
    train_dataset = ManufacturingDataset(train_indices, X_cat, X_cont, y, is_test=False)
    val_dataset = ManufacturingDataset(val_indices, X_cat, X_cont, y, is_test=False)
    test_dataset = ManufacturingDataset(test_indices, X_cat, X_cont, y, is_test=True)

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    return train_loader, val_loader, test_loader
