import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import string
from library.config import Config


def feature_engineering(df):
    """
    Applies feature engineering to the dataframe.
    1. Calculates unique_characters count from f_27.
    """
    df = df.copy()
    # Count unique characters in f_27 string
    df["unique_characters"] = df["f_27"].apply(lambda x: len(set(str(x))))
    return df


def tokenize_sequence(series, seq_len=10):
    """
    Tokenizes the f_27 string column.
    Maps A-Z to 1-26. 0 is reserved for padding/unknown.
    Returns a numpy array of shape (N, seq_len).
    """
    # Create mapping
    vocab = {char: idx + 1 for idx, char in enumerate(string.ascii_uppercase)}

    def encode(s):
        # Truncate or pad to seq_len
        s = str(s)[:seq_len]
        enc = [vocab.get(c, 0) for c in s]
        # Pad if shorter (though data is fixed length 10 usually)
        if len(enc) < seq_len:
            enc += [0] * (seq_len - len(enc))
        return enc

    # Apply encoding
    # Using list comprehension for speed over apply
    encoded = [encode(s) for s in series.values]
    return np.array(encoded, dtype=np.int64)


def preprocess_data(load_cached_data=True):
    """
    Loads data, performs feature engineering, scaling, and tokenization.
    Implements caching using .npy files.

    Returns:
        data_dict: Dictionary containing processed arrays for train, val, and test.
            keys: X_num_train, X_seq_train, y_train,
                  X_num_val, X_seq_val, y_val,
                  X_num_test, X_seq_test, ids_test
    """
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file paths
    cache_files = {
        "X_num_train": os.path.join(cache_dir, "X_num_train.npy"),
        "X_seq_train": os.path.join(cache_dir, "X_seq_train.npy"),
        "y_train": os.path.join(cache_dir, "y_train.npy"),
        "X_num_val": os.path.join(cache_dir, "X_num_val.npy"),
        "X_seq_val": os.path.join(cache_dir, "X_seq_val.npy"),
        "y_val": os.path.join(cache_dir, "y_val.npy"),
        "X_num_test": os.path.join(cache_dir, "X_num_test.npy"),
        "X_seq_test": os.path.join(cache_dir, "X_seq_test.npy"),
        "ids_test": os.path.join(cache_dir, "ids_test.npy"),
    }

    # Check if all cache files exist
    all_cached = all(os.path.exists(p) for p in cache_files.values())

    if load_cached_data and all_cached and not Config.DEBUG:
        print("Loading cached preprocessed data...")
        data = {}
        for k, v in cache_files.items():
            data[k] = np.load(v)
        return data

    print("Computing preprocessed data from scratch...")

    # Load Metadata CSVs
    # Note: The metadata CSVs contain the actual data features
    df_train = pd.read_csv(Config.TRAIN_META_PATH)
    df_val = pd.read_csv(Config.VAL_META_PATH)
    df_test = pd.read_csv(Config.TEST_META_PATH)

    # Debug mode
    if Config.DEBUG:
        print(f"DEBUG MODE: Sampling {Config.DEBUG_SAMPLES} rows.")
        df_train = df_train.iloc[: Config.DEBUG_SAMPLES]
        df_val = df_val.iloc[: Config.DEBUG_SAMPLES]
        df_test = df_test.iloc[: Config.DEBUG_SAMPLES]

    # Feature Engineering
    print("Applying feature engineering...")
    df_train = feature_engineering(df_train)
    df_val = feature_engineering(df_val)
    df_test = feature_engineering(df_test)

    # Identify Numerical Columns
    # Exclude id, target, source_path, f_27
    exclude_cols = ["id", "target", "source_path", "f_27"]
    num_cols = [c for c in df_train.columns if c not in exclude_cols]

    # Ensure consistent order
    num_cols = sorted(num_cols)
    print(f"Numerical features ({len(num_cols)}): {num_cols}")

    # Extract Numerical Data
    X_num_train = df_train[num_cols].values.astype(np.float32)
    X_num_val = df_val[num_cols].values.astype(np.float32)
    X_num_test = df_test[num_cols].values.astype(np.float32)

    # Standardization (Fit on Train, Transform All)
    scaler = StandardScaler()
    X_num_train = scaler.fit_transform(X_num_train)
    X_num_val = scaler.transform(X_num_val)
    X_num_test = scaler.transform(X_num_test)

    # Extract Sequence Data (f_27)
    print("Tokenizing sequences...")
    X_seq_train = tokenize_sequence(df_train["f_27"])
    X_seq_val = tokenize_sequence(df_val["f_27"])
    X_seq_test = tokenize_sequence(df_test["f_27"])

    # Extract Targets and IDs
    y_train = df_train["target"].values.astype(np.float32)
    y_val = df_val["target"].values.astype(np.float32)
    ids_test = df_test["id"].values.astype(np.int64)

    # Save to Cache
    print("Saving data to cache...")
    np.save(cache_files["X_num_train"], X_num_train)
    np.save(cache_files["X_seq_train"], X_seq_train)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["X_num_val"], X_num_val)
    np.save(cache_files["X_seq_val"], X_seq_val)
    np.save(cache_files["y_val"], y_val)
    np.save(cache_files["X_num_test"], X_num_test)
    np.save(cache_files["X_seq_test"], X_seq_test)
    np.save(cache_files["ids_test"], ids_test)

    return {
        "X_num_train": X_num_train,
        "X_seq_train": X_seq_train,
        "y_train": y_train,
        "X_num_val": X_num_val,
        "X_seq_val": X_seq_val,
        "y_val": y_val,
        "X_num_test": X_num_test,
        "X_seq_test": X_seq_test,
        "ids_test": ids_test,
    }


class ManufacturingDataset(Dataset):
    def __init__(self, x_num, x_seq, target=None):
        self.x_num = torch.tensor(x_num, dtype=torch.float32)
        self.x_seq = torch.tensor(x_seq, dtype=torch.long)

        if target is not None:
            self.target = torch.tensor(target, dtype=torch.float32)
        else:
            self.target = None

    def __len__(self):
        return len(self.x_num)

    def __getitem__(self, idx):
        sample = {"x_num": self.x_num[idx], "x_seq": self.x_seq[idx]}

        if self.target is not None:
            sample["target"] = self.target[idx]
        else:
            # Return NaN for unlabeled data to be handled by loss function
            sample["target"] = torch.tensor(float("nan"), dtype=torch.float32)

        return sample


def get_dataloaders(config: Config):
    """
    Creates dataloaders for training, validation, and testing.
    Implements Semi-Supervised Logic:
    - Train Loader: Concatenation of Labeled Train + Unlabeled Test
    - Val Loader: Labeled Val
    - Test Loader: Unlabeled Test (for inference)
    """
    data = preprocess_data(load_cached_data=True)

    # -------------------------------------------------------
    # Semi-Supervised Training Set Construction
    # -------------------------------------------------------
    # We combine Train and Test features for the training loop.
    # Targets for Test set are set to NaN.

    print("Constructing Semi-Supervised Training Set...")

    # Labeled Data
    x_num_labeled = data["X_num_train"]
    x_seq_labeled = data["X_seq_train"]
    y_labeled = data["y_train"]

    # Unlabeled Data (Test Set)
    x_num_unlabeled = data["X_num_test"]
    x_seq_unlabeled = data["X_seq_test"]
    y_unlabeled = np.full(len(x_num_unlabeled), np.nan, dtype=np.float32)

    # Concatenate
    x_num_combined = np.concatenate([x_num_labeled, x_num_unlabeled], axis=0)
    x_seq_combined = np.concatenate([x_seq_labeled, x_seq_unlabeled], axis=0)
    y_combined = np.concatenate([y_labeled, y_unlabeled], axis=0)

    # Create Datasets
    train_dataset = ManufacturingDataset(x_num_combined, x_seq_combined, y_combined)
    val_dataset = ManufacturingDataset(
        data["X_num_val"], data["X_seq_val"], data["y_val"]
    )
    test_dataset = ManufacturingDataset(
        data["X_num_test"], data["X_seq_test"], target=None
    )

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Train Loader: {len(train_dataset)} samples (Labeled + Unlabeled)")
    print(f"Val Loader:   {len(val_dataset)} samples")
    print(f"Test Loader:  {len(test_dataset)} samples")

    return train_loader, val_loader, test_loader
