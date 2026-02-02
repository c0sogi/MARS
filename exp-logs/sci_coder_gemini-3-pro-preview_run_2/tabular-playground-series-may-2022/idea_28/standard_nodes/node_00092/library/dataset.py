import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# Set fixed seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


class ManufacturingDataset(Dataset):
    """
    PyTorch Dataset for Manufacturing Control Data.
    Serves numerical features, categorical sequence tokens, and targets.
    """

    def __init__(self, numerical_data, categorical_data, targets=None):
        self.numerical_data = torch.FloatTensor(numerical_data)
        self.categorical_data = torch.LongTensor(categorical_data)
        self.targets = torch.FloatTensor(targets) if targets is not None else None

    def __len__(self):
        return len(self.numerical_data)

    def __getitem__(self, idx):
        sample = {
            "numerical": self.numerical_data[idx],
            "categorical": self.categorical_data[idx],
        }
        if self.targets is not None:
            sample["target"] = self.targets[idx]
        return sample


def _process_f27_sequence(series):
    """
    Decomposes the f_27 string column into a sequence of 10 integer tokens.
    Mapping: 'A' -> 0, 'B' -> 1, etc.
    """
    # Convert series of strings to list of lists of character codes
    # We assume the strings are uppercase letters.
    # ord('A') is 65. So 'A' -> 0, 'B' -> 1.
    return np.array(
        [[ord(c) - ord("A") for c in s] for s in series.values], dtype=np.int64
    )


def get_data(load_cached_data=True):
    """
    Loads, preprocesses, and caches data.

    Logic:
    1. Check if cached .npz exists in ./working/idea_28/ and load_cached_data is True.
    2. If yes, load and return.
    3. If no, load raw CSVs and Metadata.
    4. Filter and align data based on metadata IDs.
    5. Process f_27 into integer sequences.
    6. Scale numerical features (Fit on Train, Transform Val/Test).
    7. Save to cache.
    """
    cache_dir = "./working/idea_28"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "processed_data.npz")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        data = np.load(cache_path)
        return (
            data["train_num"],
            data["train_cat"],
            data["train_target"],
            data["val_num"],
            data["val_cat"],
            data["val_target"],
            data["test_num"],
            data["test_cat"],
            data["test_ids"],
        )

    print("Processing data from scratch...")

    # 1. Load Metadata
    train_meta = pd.read_csv("./metadata/train_metadata.csv")
    val_meta = pd.read_csv("./metadata/val_metadata.csv")
    test_meta = pd.read_csv("./metadata/test_metadata.csv")

    # 2. Load Raw Data
    # We load the full files and index by ID for fast lookup
    print("Reading raw CSV files...")
    df_train_raw = pd.read_csv("./input/train.csv").set_index("id")
    df_test_raw = pd.read_csv("./input/test.csv").set_index("id")

    # 3. Align Data with Metadata
    # Extract training subset
    train_ids = train_meta["id"].values
    df_train = df_train_raw.loc[train_ids]
    y_train = train_meta["target"].values.reshape(-1, 1).astype(np.float32)

    # Extract validation subset
    val_ids = val_meta["id"].values
    df_val = df_train_raw.loc[val_ids]
    y_val = val_meta["target"].values.reshape(-1, 1).astype(np.float32)

    # Extract test set
    test_ids = test_meta["id"].values
    df_test = df_test_raw.loc[test_ids]

    # 4. Feature Selection
    # f_27 is categorical, f_00..f_30 (excluding 27) are numerical
    all_cols = df_train.columns.tolist()
    num_cols = [c for c in all_cols if c.startswith("f_") and c != "f_27"]

    # 5. Process Numerical Features
    print("Scaling numerical features...")
    X_train_num = df_train[num_cols].values.astype(np.float32)
    X_val_num = df_val[num_cols].values.astype(np.float32)
    X_test_num = df_test[num_cols].values.astype(np.float32)

    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train_num)
    X_val_num = scaler.transform(X_val_num)
    X_test_num = scaler.transform(X_test_num)

    # 6. Process Categorical Feature (f_27)
    print("Tokenizing categorical sequence f_27...")
    X_train_cat = _process_f27_sequence(df_train["f_27"])
    X_val_cat = _process_f27_sequence(df_val["f_27"])
    X_test_cat = _process_f27_sequence(df_test["f_27"])

    # 7. Save to Cache
    print(f"Saving processed data to {cache_path}")
    np.savez(
        cache_path,
        train_num=X_train_num,
        train_cat=X_train_cat,
        train_target=y_train,
        val_num=X_val_num,
        val_cat=X_val_cat,
        val_target=y_val,
        test_num=X_test_num,
        test_cat=X_test_cat,
        test_ids=test_ids,
    )

    return (
        X_train_num,
        X_train_cat,
        y_train,
        X_val_num,
        X_val_cat,
        y_val,
        X_test_num,
        X_test_cat,
        test_ids,
    )


def get_dataloaders(batch_size=1024, num_workers=4, load_cached_data=True):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Returns:
        train_loader, val_loader, test_loader, test_ids
    """
    (
        train_num,
        train_cat,
        train_y,
        val_num,
        val_cat,
        val_y,
        test_num,
        test_cat,
        test_ids,
    ) = get_data(load_cached_data=load_cached_data)

    # Create Datasets
    train_dataset = ManufacturingDataset(train_num, train_cat, train_y)
    val_dataset = ManufacturingDataset(val_num, val_cat, val_y)
    test_dataset = ManufacturingDataset(test_num, test_cat, targets=None)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
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

    return train_loader, val_loader, test_loader, test_ids
