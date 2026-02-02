import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config


def process_f27(series):
    """
    Maps strings of length 10 (A-Z) to integers (1-26).
    Input: Pandas Series or Numpy array of strings.
    Output: Numpy array of shape (N, 10) with integers.
    """
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    # Map 'A' -> 1, 'B' -> 2, etc.
    char_map = {c: i + 1 for i, c in enumerate(chars)}

    # Convert series of strings to list of lists of characters
    # This is generally faster than iterating row by row in pure Python
    # We assume all strings are length 10 as per dataset description
    arr = np.array([list(s) for s in series])

    # Map characters to integers
    out = np.zeros(arr.shape, dtype=np.int64)
    for c, idx in char_map.items():
        out[arr == c] = idx

    return out


def load_and_process_data(load_cached_data=True):
    """
    Loads raw data, processes features, and caches the result.
    If cache exists and load_cached_data is True, loads from disk.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(Config.WORKING_DIR, "processed_data.npz")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        try:
            data = np.load(cache_path)
            return (
                data["X_cat_train"],
                data["X_cont_train"],
                data["y_train"],
                data["X_cat_val"],
                data["X_cont_val"],
                data["y_val"],
                data["X_cat_test"],
                data["X_cont_test"],
                data["test_ids"],
            )
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing from scratch.")

    # 2. Process from scratch
    print("Processing data from scratch...")

    # Load Metadata
    train_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, Config.TRAIN_META))
    val_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, Config.VAL_META))
    test_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, Config.TEST_META))

    # Load Raw Data
    # We read the full files and then index them using the metadata IDs
    df_train_full = pd.read_csv(os.path.join(Config.INPUT_DIR, Config.TRAIN_FILE))
    df_test_full = pd.read_csv(os.path.join(Config.INPUT_DIR, Config.TEST_FILE))

    # Index by ID for fast lookup
    df_train_full.set_index("id", inplace=True)
    df_test_full.set_index("id", inplace=True)

    # Select subsets based on metadata IDs
    # Note: Metadata IDs are integers, ensure index is too
    train_df = df_train_full.loc[train_meta["id"]]
    val_df = df_train_full.loc[val_meta["id"]]
    test_df = df_test_full.loc[test_meta["id"]]

    # Extract Targets
    y_train = train_df["target"].values.astype(np.float32)
    y_val = val_df["target"].values.astype(np.float32)

    # Feature Definition
    cat_col = "f_27"
    # Continuous columns are all columns except target and f_27
    # Note: 'id' is already the index, so it's not in columns
    all_cols = train_df.columns.tolist()
    cont_cols = [c for c in all_cols if c != "target" and c != cat_col]

    # Process Categorical Feature (f_27)
    print("Encoding categorical features...")
    X_cat_train = process_f27(train_df[cat_col].values)
    X_cat_val = process_f27(val_df[cat_col].values)
    X_cat_test = process_f27(test_df[cat_col].values)

    # Process Continuous Features (Normalization)
    print("Normalizing continuous features...")
    scaler = StandardScaler()

    # Fit only on training data
    X_cont_train = scaler.fit_transform(train_df[cont_cols].values).astype(np.float32)
    X_cont_val = scaler.transform(val_df[cont_cols].values).astype(np.float32)
    X_cont_test = scaler.transform(test_df[cont_cols].values).astype(np.float32)

    # Test IDs for submission
    test_ids = test_meta["id"].values

    # 3. Save to cache
    print(f"Saving cache to {cache_path}")
    np.savez(
        cache_path,
        X_cat_train=X_cat_train,
        X_cont_train=X_cont_train,
        y_train=y_train,
        X_cat_val=X_cat_val,
        X_cont_val=X_cont_val,
        y_val=y_val,
        X_cat_test=X_cat_test,
        X_cont_test=X_cont_test,
        test_ids=test_ids,
    )

    return (
        X_cat_train,
        X_cont_train,
        y_train,
        X_cat_val,
        X_cont_val,
        y_val,
        X_cat_test,
        X_cont_test,
        test_ids,
    )


class ManufacturingDataset(Dataset):
    def __init__(self, x_cat, x_cont, y=None):
        """
        PyTorch Dataset for the Manufacturing Task.

        Args:
            x_cat (np.ndarray): Categorical features (tokenized f_27), shape (N, 10).
            x_cont (np.ndarray): Continuous features, shape (N, 30).
            y (np.ndarray, optional): Target labels, shape (N,).
        """
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        self.x_cont = torch.tensor(x_cont, dtype=torch.float32)

        if y is not None:
            # BCEWithLogitsLoss expects float targets, usually shape (N, 1)
            self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        else:
            self.y = None

    def __len__(self):
        return len(self.x_cat)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.x_cat[idx], self.x_cont[idx], self.y[idx]
        return self.x_cat[idx], self.x_cont[idx]


def get_dataloaders(load_cached_data=True, batch_size=None, num_workers=None):
    """
    Factory function to create DataLoaders for Train, Val, and Test.

    Args:
        load_cached_data (bool): Whether to use cached numpy arrays.
        batch_size (int, optional): Override batch size from Config.
        num_workers (int, optional): Override num_workers from Config.

    Returns:
        train_loader, val_loader, test_loader, test_ids
    """
    # Use config defaults if not provided
    bs = batch_size if batch_size is not None else Config.BATCH_SIZE
    nw = num_workers if num_workers is not None else Config.NUM_WORKERS

    # Load data
    data = load_and_process_data(load_cached_data=load_cached_data)
    (
        X_cat_train,
        X_cont_train,
        y_train,
        X_cat_val,
        X_cont_val,
        y_val,
        X_cat_test,
        X_cont_test,
        test_ids,
    ) = data

    # Create Datasets
    train_dataset = ManufacturingDataset(X_cat_train, X_cont_train, y_train)
    val_dataset = ManufacturingDataset(X_cat_val, X_cont_val, y_val)
    test_dataset = ManufacturingDataset(X_cat_test, X_cont_test, y=None)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True
    )

    return train_loader, val_loader, test_loader, test_ids
