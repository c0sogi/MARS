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
    """

    def __init__(self, continuous_data, categorical_data, targets=None):
        self.continuous_data = torch.FloatTensor(continuous_data)
        self.categorical_data = torch.LongTensor(categorical_data)
        self.targets = torch.FloatTensor(targets) if targets is not None else None

    def __len__(self):
        return len(self.continuous_data)

    def __getitem__(self, idx):
        item = {
            "continuous": self.continuous_data[idx],
            "categorical": self.categorical_data[idx],
        }
        if self.targets is not None:
            item["target"] = self.targets[idx]
        return item


def process_data(load_cached_data=True):
    """
    Loads raw data, performs feature engineering (tokenization, normalization),
    and caches the result to disk.

    Args:
        load_cached_data (bool): If True, attempts to load from existing .npz file.

    Returns:
        dict: A dictionary containing processed numpy arrays for train/test splits.
    """
    cache_path = Config.PROCESSED_DATA_PATH

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            data = np.load(cache_path)
            return {
                "train_cont": data["train_cont"],
                "train_cat": data["train_cat"],
                "train_target": data["train_target"],
                "train_ids": data["train_ids"],
                "test_cont": data["test_cont"],
                "test_cat": data["test_cat"],
                "test_ids": data["test_ids"],
            }
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing data...")

    # 2. Process from scratch
    print("Processing data from scratch...")

    # Load raw CSVs
    train_df = pd.read_csv(Config.TRAIN_DATA_PATH)
    test_df = pd.read_csv(Config.TEST_DATA_PATH)

    # Identify columns
    # Continuous: f_00 to f_30 excluding f_27
    # Categorical: f_27
    feature_cols = [c for c in train_df.columns if c.startswith("f_") and c != "f_27"]
    cat_col = "f_27"

    # --- Process Categorical (f_27) ---
    # Map A-Z to 1-26.
    # We assume standard uppercase letters based on dataset description.
    def encode_sequence(seq_series):
        # seq_series is a pandas Series of strings
        # Convert to list of lists of ordinals
        # 'A' is 65. We want 'A' -> 1. So ord(c) - 64.
        # This is vectorized by converting to a char array view if fixed width,
        # but list comprehension is robust enough for 1M rows.
        return np.array([[ord(c) - 64 for c in s] for s in seq_series], dtype=np.int64)

    print("Encoding categorical sequences...")
    X_train_cat = encode_sequence(train_df[cat_col])
    X_test_cat = encode_sequence(test_df[cat_col])

    # --- Process Continuous ---
    print("Normalizing continuous features...")
    scaler = StandardScaler()

    # Fit on train, transform both
    X_train_cont = scaler.fit_transform(
        train_df[feature_cols].values.astype(np.float32)
    )
    X_test_cont = scaler.transform(test_df[feature_cols].values.astype(np.float32))

    # --- Extract Targets and IDs ---
    y_train = train_df["target"].values.astype(np.float32)
    ids_train = train_df["id"].values.astype(np.int64)
    ids_test = test_df["id"].values.astype(np.int64)

    # --- Save to Cache ---
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        train_cont=X_train_cont,
        train_cat=X_train_cat,
        train_target=y_train,
        train_ids=ids_train,
        test_cont=X_test_cont,
        test_cat=X_test_cat,
        test_ids=ids_test,
    )
    print(f"Data processed and saved to {cache_path}")

    return {
        "train_cont": X_train_cont,
        "train_cat": X_train_cat,
        "train_target": y_train,
        "train_ids": ids_train,
        "test_cont": X_test_cont,
        "test_cat": X_test_cat,
        "test_ids": ids_test,
    }


def get_dataloaders(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, load_cached_data=True
):
    """
    Creates DataLoaders for Train, Validation, and Test sets using metadata splits.

    Args:
        batch_size (int): Batch size for the dataloaders.
        num_workers (int): Number of worker threads.
        load_cached_data (bool): Whether to use cached preprocessed data.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 1. Get processed full datasets
    data = process_data(load_cached_data=load_cached_data)

    # 2. Load Metadata for Splitting
    print("Loading metadata for splits...")
    train_meta = pd.read_csv(Config.TRAIN_META_PATH)
    val_meta = pd.read_csv(Config.VAL_META_PATH)
    test_meta = pd.read_csv(Config.TEST_META_PATH)

    # 3. Map IDs to Array Indices
    # The processed data arrays (from process_data) correspond to the rows in the raw files.
    # We need to map the ID to the row index to select the correct subset.

    # Create lookup tables: ID -> Index
    train_id_map = {uid: i for i, uid in enumerate(data["train_ids"])}
    test_id_map = {uid: i for i, uid in enumerate(data["test_ids"])}

    def get_indices(meta_df, id_map, name="split"):
        indices = []
        missing = 0
        for uid in meta_df["id"].values:
            if uid in id_map:
                indices.append(id_map[uid])
            else:
                missing += 1
        if missing > 0:
            print(
                f"Warning: {missing} IDs from {name} metadata not found in processed data."
            )
        return np.array(indices)

    train_indices = get_indices(train_meta, train_id_map, "train")
    val_indices = get_indices(val_meta, train_id_map, "val")
    test_indices = get_indices(test_meta, test_id_map, "test")

    # 4. Create Datasets
    train_dataset = ManufacturingDataset(
        continuous_data=data["train_cont"][train_indices],
        categorical_data=data["train_cat"][train_indices],
        targets=data["train_target"][train_indices],
    )

    val_dataset = ManufacturingDataset(
        continuous_data=data["train_cont"][val_indices],
        categorical_data=data["train_cat"][val_indices],
        targets=data["train_target"][val_indices],
    )

    test_dataset = ManufacturingDataset(
        continuous_data=data["test_cont"][test_indices],
        categorical_data=data["test_cat"][test_indices],
        targets=None,  # No targets for test set
    )

    print(
        f"Dataset Sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    # 5. Create Loaders
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
