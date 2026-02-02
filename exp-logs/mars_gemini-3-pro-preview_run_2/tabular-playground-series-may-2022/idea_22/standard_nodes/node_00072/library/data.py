import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.utils import seed_everything


# ------------------------------------------------------------------------------
# Dataset Class
# ------------------------------------------------------------------------------
class ManufacturingDataset(Dataset):
    def __init__(self, cont_features, seq_features, targets=None):
        """
        PyTorch Dataset for the manufacturing task.

        Args:
            cont_features (np.ndarray): Normalized continuous features (N, 30).
            seq_features (np.ndarray): Integer-encoded sequence features (N, 10).
            targets (np.ndarray, optional): Binary targets (N,).
        """
        self.cont_features = cont_features
        self.seq_features = seq_features
        self.targets = targets

    def __len__(self):
        return len(self.cont_features)

    def __getitem__(self, idx):
        item = {
            "cont": torch.tensor(self.cont_features[idx], dtype=torch.float32),
            "seq": torch.tensor(self.seq_features[idx], dtype=torch.long),
        }

        if self.targets is not None:
            item["target"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        return item


# ------------------------------------------------------------------------------
# Data Processing Logic
# ------------------------------------------------------------------------------
def _vectorize_sequence(series):
    """
    Converts a pandas Series of strings (length 10) into a (N, 10) numpy array of integers.
    Mapping: 'A' -> 1, 'B' -> 2, ...
    """
    # Convert series to list of strings
    str_list = series.tolist()
    # Pre-allocate array
    n = len(str_list)
    seq_len = 10
    # Map 'A' (65) to 1
    offset = ord("A") - 1

    # Efficient vectorization using list comprehension and numpy
    # Assuming all strings are length 10 and uppercase letters
    # We can cast to uint8 (ASCII) and subtract offset
    # But doing it explicitly is safer for mixed types, though dataset is clean.
    # Fast approach:
    # Convert to bytearray, reshape.
    # This works if all strings are ascii and same length.
    # Fallback to list comp for safety:
    data = [[ord(c) - offset for c in s] for s in str_list]
    return np.array(data, dtype=np.int64)


def process_data(load_cached_data=True):
    """
    Loads raw data, performs feature engineering (scaling, tokenization),
    aligns with metadata splits, and caches the result.

    Args:
        load_cached_data (bool): If True, attempts to load from disk.

    Returns:
        dict: Dictionary containing numpy arrays for train/val/test splits.
    """
    cache_path = Config.PROCESSED_DATA_PATH

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            data = np.load(cache_path)
            return dict(data)
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    print("Processing data from scratch...")

    # 2. Load Metadata
    train_meta = pd.read_csv(Config.TRAIN_META)
    val_meta = pd.read_csv(Config.VAL_META)
    test_meta = pd.read_csv(Config.TEST_META)

    # 3. Load Raw Data
    # We load the full train.csv and test.csv
    # Note: train.csv contains both train and val samples
    df_train_full = pd.read_csv(Config.TRAIN_CSV)
    df_test = pd.read_csv(Config.TEST_CSV)

    # 4. Feature Identification
    # Continuous features: f_00 to f_30, excluding f_27
    # We can identify them by column name pattern or exclusion
    all_cols = df_train_full.columns.tolist()
    cont_cols = [c for c in all_cols if c not in ["id", "target", "f_27"]]

    # Verify count matches config expectation (optional but good practice)
    assert (
        len(cont_cols) == Config.NUM_CONT_FEATURES
    ), f"Expected {Config.NUM_CONT_FEATURES} continuous features, found {len(cont_cols)}"

    # 5. Preprocessing - Continuous Features (StandardScaler)
    scaler = StandardScaler()

    # IMPORTANT: Fit ONLY on the training subset to avoid leakage
    # Get IDs strictly belonging to the training set
    train_ids_set = set(train_meta["id"].values)

    # Create a mask for rows in df_train_full that are in the training set
    train_mask = df_train_full["id"].isin(train_ids_set)

    # Fit scaler
    scaler.fit(df_train_full.loc[train_mask, cont_cols])

    # Transform all
    # We transform the full train dataframe first, then split later
    full_train_cont_scaled = scaler.transform(df_train_full[cont_cols])
    test_cont_scaled = scaler.transform(df_test[cont_cols])

    # 6. Preprocessing - Sequence Feature (f_27)
    full_train_seq = _vectorize_sequence(df_train_full["f_27"])
    test_seq = _vectorize_sequence(df_test["f_27"])

    # 7. Align with Metadata Splits
    # We need to map ID -> Index in the loaded dataframes to extract correct rows

    # Create lookup for train.csv
    # id_to_idx maps the sample ID to its row index in df_train_full / arrays
    train_id_to_idx = pd.Series(
        df_train_full.index.values, index=df_train_full["id"]
    ).to_dict()

    # Extract Train Indices
    train_indices = [train_id_to_idx[uid] for uid in train_meta["id"]]
    # Extract Val Indices
    val_indices = [train_id_to_idx[uid] for uid in val_meta["id"]]

    # Create final arrays
    X_train_cont = full_train_cont_scaled[train_indices]
    X_train_seq = full_train_seq[train_indices]
    y_train = train_meta["target"].values.astype(np.float32)  # Metadata has target

    X_val_cont = full_train_cont_scaled[val_indices]
    X_val_seq = full_train_seq[val_indices]
    y_val = val_meta["target"].values.astype(np.float32)

    # For test, we align with test_metadata IDs
    # Though usually test.csv is already in order, we enforce it
    test_id_to_idx = pd.Series(df_test.index.values, index=df_test["id"]).to_dict()
    test_indices = [test_id_to_idx[uid] for uid in test_meta["id"]]

    X_test_cont = test_cont_scaled[test_indices]
    X_test_seq = test_seq[test_indices]
    test_ids = test_meta["id"].values

    # 8. Save to Cache
    data_dict = {
        "train_cont": X_train_cont,
        "train_seq": X_train_seq,
        "train_target": y_train,
        "val_cont": X_val_cont,
        "val_seq": X_val_seq,
        "val_target": y_val,
        "test_cont": X_test_cont,
        "test_seq": X_test_seq,
        "test_ids": test_ids,
    }

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, **data_dict)
    print(f"Data processed and saved to {cache_path}")

    return data_dict


# ------------------------------------------------------------------------------
# DataLoader Factory
# ------------------------------------------------------------------------------
def get_dataloaders(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, load_cached_data=True
):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.
        load_cached_data (bool): Whether to use cached preprocessed data.

    Returns:
        dict: Dictionary containing 'train', 'val', 'test' DataLoaders.
    """
    seed_everything(Config.RANDOM_STATE)

    # Get Data
    data = process_data(load_cached_data=load_cached_data)

    # Create Datasets
    train_ds = ManufacturingDataset(
        data["train_cont"], data["train_seq"], data["train_target"]
    )

    val_ds = ManufacturingDataset(data["val_cont"], data["val_seq"], data["val_target"])

    test_ds = ManufacturingDataset(data["test_cont"], data["test_seq"], targets=None)

    # Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Helpful for BatchNorm stats stability
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "test_ids": data["test_ids"],  # Return IDs for submission creation
    }
