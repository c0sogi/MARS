import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from library.config import Config


def tokenize_f27(series):
    """
    Decomposes the f_27 string feature into integer tokens.
    Maps 'A'->1, 'B'->2, ..., 'Z'->26.
    """
    # Ensure series is string type
    series = series.astype(str)

    # Vectorized conversion: ord(c) - 64 maps 'A' (65) to 1
    # We iterate over the series, and for each string, create a list of char codes.
    tokenized = [[ord(c) - 64 for c in s] for s in series]

    return np.array(tokenized, dtype=np.int64)


def load_and_preprocess_data(load_cached_data=True):
    """
    Loads raw data, applies preprocessing (scaling, tokenization),
    and caches the result to disk.

    Returns:
        dict: A dictionary containing processed numpy arrays for all splits.
    """
    cache_path = Config.CACHE_PATH

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        try:
            # Load into memory as a dict to ensure file handle is closed and data is accessible
            data = dict(np.load(cache_path))
            # Basic validation of keys
            if "X_cont_train" in data:
                return data
            else:
                print("Cache invalid (missing keys). Recomputing...")
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing...")

    # 2. Process from scratch
    print("Processing data from scratch...")

    # Load Metadata
    train_meta = pd.read_csv(Config.TRAIN_META)
    val_meta = pd.read_csv(Config.VAL_META)
    test_meta = pd.read_csv(Config.TEST_META)

    # Load Raw Data
    # train.csv contains data for both train and val splits
    df_train_raw = pd.read_csv(Config.TRAIN_CSV)
    df_test_raw = pd.read_csv(Config.TEST_CSV)

    # Index by ID for fast lookup during merge
    df_train_raw = df_train_raw.set_index("id")
    df_test_raw = df_test_raw.set_index("id")

    # Define Feature Columns
    # f_00 to f_30 are continuous, except f_27 which is categorical
    cont_cols = [f"f_{i:02d}" for i in range(31) if i != 27]
    cat_col = "f_27"

    def extract_split(meta_df, raw_df, is_test=False):
        """Helper to extract features based on metadata IDs."""
        ids = meta_df["id"].values

        # Select rows matching IDs from the raw dataframe
        # Using .loc with index is efficient
        subset = raw_df.loc[ids]

        # Continuous Features
        X_cont = subset[cont_cols].values.astype(np.float32)

        # Categorical Feature (Tokenized)
        X_cat = tokenize_f27(subset[cat_col])

        # Target
        if not is_test:
            # Prefer target from metadata to ensure strict alignment with stratified split
            y = meta_df["target"].values.astype(np.float32)
        else:
            y = np.zeros(len(ids), dtype=np.float32)

        return X_cont, X_cat, y

    # Extract Data
    print("Extracting Train split...")
    X_cont_train, X_cat_train, y_train = extract_split(train_meta, df_train_raw)

    print("Extracting Validation split...")
    X_cont_val, X_cat_val, y_val = extract_split(val_meta, df_train_raw)

    print("Extracting Test split...")
    X_cont_test, X_cat_test, _ = extract_split(test_meta, df_test_raw, is_test=True)

    # Scaling
    # Fit StandardScaler ONLY on training data
    print("Fitting StandardScaler on Train data...")
    scaler = StandardScaler()
    X_cont_train = scaler.fit_transform(X_cont_train)

    # Transform Validation and Test data using the train-fitted scaler
    X_cont_val = scaler.transform(X_cont_val)
    X_cont_test = scaler.transform(X_cont_test)

    # Save to Cache
    print(f"Saving processed data to {cache_path}...")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(
        cache_path,
        X_cont_train=X_cont_train,
        X_cat_train=X_cat_train,
        y_train=y_train,
        X_cont_val=X_cont_val,
        X_cat_val=X_cat_val,
        y_val=y_val,
        X_cont_test=X_cont_test,
        X_cat_test=X_cat_test,
    )

    # Return dictionary
    return {
        "X_cont_train": X_cont_train,
        "X_cat_train": X_cat_train,
        "y_train": y_train,
        "X_cont_val": X_cont_val,
        "X_cat_val": X_cat_val,
        "y_val": y_val,
        "X_cont_test": X_cont_test,
        "X_cat_test": X_cat_test,
    }


class ManufacturingDataset(Dataset):
    def __init__(self, split="train", load_cached_data=True):
        """
        PyTorch Dataset for the Manufacturing task.

        Args:
            split (str): One of 'train', 'val', 'test'.
            load_cached_data (bool): Whether to use cached preprocessed data.
        """
        super().__init__()
        self.split = split

        # Load all data (either from cache or by processing raw files)
        data = load_and_preprocess_data(load_cached_data=load_cached_data)

        # Assign arrays based on the requested split
        if split == "train":
            self.X_cont = data["X_cont_train"]
            self.X_cat = data["X_cat_train"]
            self.y = data["y_train"]
        elif split == "val":
            self.X_cont = data["X_cont_val"]
            self.X_cat = data["X_cat_val"]
            self.y = data["y_val"]
        elif split == "test":
            self.X_cont = data["X_cont_test"]
            self.X_cat = data["X_cat_test"]
            self.y = None
        else:
            raise ValueError(
                f"Invalid split '{split}'. Must be 'train', 'val', or 'test'."
            )

        # Convert NumPy arrays to PyTorch Tensors
        self.X_cont = torch.from_numpy(self.X_cont).float()
        self.X_cat = torch.from_numpy(self.X_cat).long()

        if self.y is not None:
            # Reshape target to (N, 1) for BCEWithLogitsLoss
            self.y = torch.from_numpy(self.y).float().unsqueeze(1)

    def __len__(self):
        return len(self.X_cont)

    def __getitem__(self, idx):
        """
        Returns:
            tuple: (continuous_features, categorical_features, target)
            If split is 'test', returns (continuous_features, categorical_features)
        """
        if self.y is not None:
            return self.X_cont[idx], self.X_cat[idx], self.y[idx]
        else:
            return self.X_cont[idx], self.X_cat[idx]
