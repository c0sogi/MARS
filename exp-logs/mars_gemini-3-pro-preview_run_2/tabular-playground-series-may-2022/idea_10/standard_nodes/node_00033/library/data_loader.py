import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.utils import seed_everything


class ManufacturingDataset(Dataset):
    """
    PyTorch Dataset for the Manufacturing Control task.
    Serves continuous features, tokenized categorical features, and targets.
    """

    def __init__(self, cont_features, cat_features, targets=None):
        self.cont_features = torch.FloatTensor(cont_features)
        self.cat_features = torch.LongTensor(cat_features)
        self.targets = torch.FloatTensor(targets) if targets is not None else None

    def __len__(self):
        return len(self.cont_features)

    def __getitem__(self, idx):
        item = {"cont": self.cont_features[idx], "cat": self.cat_features[idx]}
        if self.targets is not None:
            item["target"] = self.targets[idx]
        return item


def _tokenize_f27(series):
    """
    Converts a pandas Series of strings (f_27) into a numpy array of integers.
    Mapping: 'A' -> 1, 'B' -> 2, ..., 'Z' -> 26.
    """
    # Convert series to list of strings
    strings = series.values.astype(str)
    # Create a buffer to hold the integer codes.
    # Assuming fixed length of 10 as per config/EDA.
    n_samples = len(strings)
    seq_len = Config.CAT_SEQ_LEN

    # Vectorized approach: view buffer as uint8 (ASCII)
    # This works efficiently because f_27 is ASCII.
    # We create a char array
    char_array = np.array([list(s) for s in strings])

    # Map characters to integers: ord(c) - ord('A') + 1
    # 'A' is 65. So we subtract 64.
    tokenized = np.vectorize(lambda x: ord(x) - 64)(char_array)

    return tokenized.astype(np.int64)


def process_data(load_cached_data=True):
    """
    Loads raw data, performs preprocessing (normalization, tokenization),
    splits based on metadata, and caches the result.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 1. Check Cache
    if load_cached_data and os.path.exists(Config.CACHE_FILE):
        print(f"Loading cached data from {Config.CACHE_FILE}...")
        try:
            data = np.load(Config.CACHE_FILE)
            return {
                "train_cont": data["train_cont"],
                "train_cat": data["train_cat"],
                "train_y": data["train_y"],
                "val_cont": data["val_cont"],
                "val_cat": data["val_cat"],
                "val_y": data["val_y"],
                "test_cont": data["test_cont"],
                "test_cat": data["test_cat"],
                "test_ids": data["test_ids"],
            }
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    print("Processing data from scratch...")

    # 2. Load Metadata
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)
    test_meta = pd.read_csv(Config.TEST_METADATA_PATH)

    # 3. Load Raw Data
    # We load full train and test files.
    # Note: train.csv contains data for both train_meta and val_meta.
    df_train_full = pd.read_csv(Config.TRAIN_DATA_PATH)
    df_test = pd.read_csv(Config.TEST_DATA_PATH)

    # 4. Feature Selection
    cont_cols = Config.CONT_FEATURES
    cat_col = Config.CAT_FEATURE

    # 5. Continuous Feature Normalization
    # Fit scaler ONLY on the training subset defined by metadata
    train_ids = set(train_meta["id"].values)

    # Identify rows in df_train_full that belong to the training set
    train_mask = df_train_full["id"].isin(train_ids)

    scaler = StandardScaler()

    # Fit on training subset
    scaler.fit(df_train_full.loc[train_mask, cont_cols])

    # Transform everything
    train_full_cont = scaler.transform(df_train_full[cont_cols])
    test_cont = scaler.transform(df_test[cont_cols])

    # 6. Categorical Tokenization
    train_full_cat = _tokenize_f27(df_train_full[cat_col])
    test_cat = _tokenize_f27(df_test[cat_col])

    # 7. Prepare Final Arrays using Metadata alignment
    # We need to extract the specific rows for Train and Val based on IDs.
    # To do this efficiently, we can index the processed arrays by ID.

    # Create a mapping from ID to index for the full train file
    id_to_idx_train = {
        id_val: idx for idx, id_val in enumerate(df_train_full["id"].values)
    }

    # Get indices for train and val sets
    train_indices = [id_to_idx_train[i] for i in train_meta["id"].values]
    val_indices = [id_to_idx_train[i] for i in val_meta["id"].values]

    # Slice arrays
    X_train_cont = train_full_cont[train_indices]
    X_train_cat = train_full_cat[train_indices]
    y_train = train_meta["target"].values.astype(np.float32)

    X_val_cont = train_full_cont[val_indices]
    X_val_cat = train_full_cat[val_indices]
    y_val = val_meta["target"].values.astype(np.float32)

    # Test data is already separate, but we ensure order matches metadata if necessary.
    # The test metadata just lists IDs in test.csv, usually in order, but let's be safe.
    id_to_idx_test = {id_val: idx for idx, id_val in enumerate(df_test["id"].values)}
    test_indices = [id_to_idx_test[i] for i in test_meta["id"].values]

    X_test_cont = test_cont[test_indices]
    X_test_cat = test_cat[test_indices]
    test_ids = test_meta["id"].values

    # 8. Save to Cache
    data_dict = {
        "train_cont": X_train_cont,
        "train_cat": X_train_cat,
        "train_y": y_train,
        "val_cont": X_val_cont,
        "val_cat": X_val_cat,
        "val_y": y_val,
        "test_cont": X_test_cont,
        "test_cat": X_test_cat,
        "test_ids": test_ids,
    }

    np.savez(Config.CACHE_FILE, **data_dict)
    print(f"Data processed and saved to {Config.CACHE_FILE}")

    return data_dict


def get_dataloaders(debug=False, load_cached_data=True):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        debug (bool): If True, subsets the data to Config.MAX_DEBUG_SAMPLES.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        train_loader, val_loader, test_loader
    """
    seed_everything(Config.SEED)

    data = process_data(load_cached_data=load_cached_data)

    train_cont = data["train_cont"]
    train_cat = data["train_cat"]
    train_y = data["train_y"]

    val_cont = data["val_cont"]
    val_cat = data["val_cat"]
    val_y = data["val_y"]

    test_cont = data["test_cont"]
    test_cat = data["test_cat"]
    # Test has no targets

    # Handle Debug Mode
    if debug:
        print(f"Debug mode enabled. Subsampling to {Config.MAX_DEBUG_SAMPLES} samples.")
        train_cont = train_cont[: Config.MAX_DEBUG_SAMPLES]
        train_cat = train_cat[: Config.MAX_DEBUG_SAMPLES]
        train_y = train_y[: Config.MAX_DEBUG_SAMPLES]

        val_cont = val_cont[: Config.MAX_DEBUG_SAMPLES]
        val_cat = val_cat[: Config.MAX_DEBUG_SAMPLES]
        val_y = val_y[: Config.MAX_DEBUG_SAMPLES]

        # Keep test full or slice? Usually slice for debug speed
        test_cont = test_cont[: Config.MAX_DEBUG_SAMPLES]
        test_cat = test_cat[: Config.MAX_DEBUG_SAMPLES]

    # Create Datasets
    train_dataset = ManufacturingDataset(train_cont, train_cat, train_y)
    val_dataset = ManufacturingDataset(val_cont, val_cat, val_y)
    test_dataset = ManufacturingDataset(test_cont, test_cat, targets=None)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Useful for batch norm stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
