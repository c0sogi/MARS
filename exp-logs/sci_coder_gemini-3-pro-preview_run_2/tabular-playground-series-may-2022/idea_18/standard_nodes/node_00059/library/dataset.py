import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything


class ManufacturingDataset(Dataset):
    """
    PyTorch Dataset for the Manufacturing Control task.
    Handles both continuous features and the tokenized categorical sequence.
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


def _tokenize_f27(series):
    """
    Tokenizes the 'f_27' string column into a sequence of integers.
    Maps 'A'->1, 'B'->2, ..., 'Z'->26. 0 is used for padding/unknown.
    """

    def map_char(c):
        # A=65. We want A=1.
        val = ord(c) - 64
        # Clamp to ensure we stay within vocab range (0 reserved)
        return max(1, min(val, Config.VOCAB_SIZE - 1))

    tokenized = []
    for s in series:
        if not isinstance(s, str):
            # Fallback for non-string (should not happen based on EDA)
            tokens = [0] * Config.SEQ_LEN
        else:
            tokens = [map_char(c) for c in s]
            # Pad or truncate to fixed sequence length
            if len(tokens) < Config.SEQ_LEN:
                tokens += [0] * (Config.SEQ_LEN - len(tokens))
            else:
                tokens = tokens[: Config.SEQ_LEN]
        tokenized.append(tokens)

    return np.array(tokenized, dtype=np.int32)


def get_data(load_cached_data=True):
    """
    Loads raw data, performs preprocessing (normalization, tokenization),
    and caches the result to disk.

    Returns:
        tuple: (train_dict, val_dict, test_dict) containing numpy arrays.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "processed_data.npz")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            data = np.load(cache_path)
            return (
                {
                    "cont": data["train_cont"],
                    "cat": data["train_cat"],
                    "target": data["train_target"],
                },
                {
                    "cont": data["val_cont"],
                    "cat": data["val_cat"],
                    "target": data["val_target"],
                },
                {
                    "cont": data["test_cont"],
                    "cat": data["test_cat"],
                    "ids": data["test_ids"],
                },
            )
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    print("Processing data from scratch...")

    # 2. Load Metadata
    train_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "train_metadata.csv"))
    val_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "val_metadata.csv"))
    test_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "test_metadata.csv"))

    # 3. Load Raw Data
    df_train_raw = pd.read_csv(os.path.join(Config.INPUT_DIR, "train.csv"))
    df_test_raw = pd.read_csv(os.path.join(Config.INPUT_DIR, "test.csv"))

    # Index by ID for alignment with metadata
    df_train_raw.set_index("id", inplace=True)
    df_test_raw.set_index("id", inplace=True)

    # 4. Identify Columns
    # Continuous: f_00 to f_30, excluding f_27
    cont_cols = [f"f_{i:02d}" for i in range(31) if i != 27]
    cat_col = "f_27"

    # 5. Extract Splits
    # Use metadata IDs to select correct rows
    X_train_raw = df_train_raw.loc[train_meta["id"]]
    X_val_raw = df_train_raw.loc[val_meta["id"]]
    X_test_raw = df_test_raw.loc[test_meta["id"]]

    y_train = train_meta["target"].values.astype(np.float32)
    y_val = val_meta["target"].values.astype(np.float32)

    # 6. Preprocessing

    # A. Normalize Continuous Features (Z-Score)
    # Fit ONLY on training set
    print("Normalizing continuous features...")
    train_vals = X_train_raw[cont_cols].values.astype(np.float32)
    mean = np.mean(train_vals, axis=0)
    std = np.std(train_vals, axis=0)
    std = np.where(std == 0, 1.0, std)  # Prevent div by zero

    def normalize(df_subset):
        vals = df_subset[cont_cols].values.astype(np.float32)
        return (vals - mean) / std

    X_train_cont = normalize(X_train_raw)
    X_val_cont = normalize(X_val_raw)
    X_test_cont = normalize(X_test_raw)

    # B. Tokenize Categorical Feature
    print("Tokenizing categorical sequence...")
    X_train_cat = _tokenize_f27(X_train_raw[cat_col])
    X_val_cat = _tokenize_f27(X_val_raw[cat_col])
    X_test_cat = _tokenize_f27(X_test_raw[cat_col])

    test_ids = test_meta["id"].values

    # 7. Save to Cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    np.savez(
        cache_path,
        train_cont=X_train_cont,
        train_cat=X_train_cat,
        train_target=y_train,
        val_cont=X_val_cont,
        val_cat=X_val_cat,
        val_target=y_val,
        test_cont=X_test_cont,
        test_cat=X_test_cat,
        test_ids=test_ids,
    )
    print(f"Data processed and saved to {cache_path}")

    return (
        {"cont": X_train_cont, "cat": X_train_cat, "target": y_train},
        {"cont": X_val_cont, "cat": X_val_cat, "target": y_val},
        {"cont": X_test_cont, "cat": X_test_cat, "ids": test_ids},
    )


def get_dataloaders(
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
    debug=Config.DEBUG,
):
    """
    Generates PyTorch DataLoaders for Train, Validation, and Test sets.

    Args:
        batch_size (int): Batch size for loading.
        num_workers (int): Number of subprocesses for data loading.
        load_cached_data (bool): Whether to attempt loading from disk cache.
        debug (bool): If True, truncates dataset to Config.DEBUG_SAMPLES.

    Returns:
        tuple: (train_loader, val_loader, test_loader, test_ids)
    """
    train_data, val_data, test_data = get_data(load_cached_data=load_cached_data)

    # Apply Debug Limit
    if debug:
        limit = Config.DEBUG_SAMPLES
        print(f"DEBUG MODE: Limiting datasets to {limit} samples.")

        train_data["cont"] = train_data["cont"][:limit]
        train_data["cat"] = train_data["cat"][:limit]
        train_data["target"] = train_data["target"][:limit]

        val_data["cont"] = val_data["cont"][:limit]
        val_data["cat"] = val_data["cat"][:limit]
        val_data["target"] = val_data["target"][:limit]

        test_data["cont"] = test_data["cont"][:limit]
        test_data["cat"] = test_data["cat"][:limit]
        test_data["ids"] = test_data["ids"][:limit]

    # Instantiate Datasets
    train_dataset = ManufacturingDataset(
        train_data["cont"], train_data["cat"], train_data["target"]
    )

    val_dataset = ManufacturingDataset(
        val_data["cont"], val_data["cat"], val_data["target"]
    )

    test_dataset = ManufacturingDataset(
        test_data["cont"], test_data["cat"], targets=None
    )

    # Instantiate Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(Config.DEVICE == "cuda"),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    return train_loader, val_loader, test_loader, test_data["ids"]
