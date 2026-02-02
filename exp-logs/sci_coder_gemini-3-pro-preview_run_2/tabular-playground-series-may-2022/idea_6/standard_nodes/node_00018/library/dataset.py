import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from library.config import Config


class CharTokenizer:
    """
    Tokenizes the f_27 string feature into integer sequences.
    Maps characters 'A' through 'Z' to integers 1 through 26.
    0 is reserved for padding/unknown (though not expected in this clean dataset).
    """

    def __init__(self):
        self.vocab_size = Config.VOCAB_SIZE
        self.seq_len = Config.F27_SEQ_LEN

    def transform(self, series: pd.Series) -> np.ndarray:
        """
        Converts a pandas Series of strings into a numpy array of shape (N, 10).
        """
        # Vectorized implementation using list comprehension for speed
        # ord('A') is 65. We want 'A' -> 1. So ord(c) - 64.
        # We assume the input strings are clean and contain only A-Z.

        # Convert series to list of strings
        strings = series.tolist()

        # Map characters to integers
        # This list comprehension is generally faster than pandas apply for string operations
        tokenized = []
        for s in strings:
            # Truncate string to seq_len first, then convert
            t = [ord(c) - 64 for c in s[: self.seq_len]]
            # Pad with 0 if shorter than seq_len
            if len(t) < self.seq_len:
                t += [0] * (self.seq_len - len(t))
            tokenized.append(t)

        return np.array(tokenized, dtype=np.int32)


class ManufacturingDataset(Dataset):
    """
    PyTorch Dataset for the Manufacturing Control task.
    Handles continuous features, tokenized categorical features, and targets.
    """

    def __init__(self, cont_features, cat_features, targets=None):
        """
        Args:
            cont_features (np.ndarray): Normalized continuous features (N, 30).
            cat_features (np.ndarray): Tokenized categorical features (N, 10).
            targets (np.ndarray, optional): Binary targets (N,).
        """
        self.cont_features = torch.tensor(cont_features, dtype=torch.float32)
        self.cat_features = torch.tensor(cat_features, dtype=torch.long)

        if targets is not None:
            self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(
                1
            )  # (N, 1)
        else:
            self.targets = None

    def __len__(self):
        return len(self.cont_features)

    def __getitem__(self, idx):
        item = {"cont": self.cont_features[idx], "cat": self.cat_features[idx]}
        if self.targets is not None:
            item["target"] = self.targets[idx]
        return item


def get_datasets(load_cached_data=True, debug=Config.DEBUG):
    """
    Prepares and returns the Train, Validation, and Test datasets.
    Implements caching logic to avoid re-processing raw data.

    Args:
        load_cached_data (bool): If True, attempts to load from disk.
        debug (bool): If True, subsets the data for rapid prototyping.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    cache_path = Config.PROCESSED_DATA_PATH

    # --------------------------------------------------------------------------
    # 1. Attempt to Load from Cache
    # --------------------------------------------------------------------------
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            data = np.load(cache_path)
            train_cont = data["train_cont"]
            train_cat = data["train_cat"]
            train_target = data["train_target"]

            val_cont = data["val_cont"]
            val_cat = data["val_cat"]
            val_target = data["val_target"]

            test_cont = data["test_cont"]
            test_cat = data["test_cat"]

            print("Cache loaded successfully.")

        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing data...")
            load_cached_data = False  # Fallback to processing

    # --------------------------------------------------------------------------
    # 2. Process from Scratch (if cache missing or forced)
    # --------------------------------------------------------------------------
    if not load_cached_data or not os.path.exists(cache_path):
        print("Processing raw data...")

        # Load Metadata
        train_meta = pd.read_csv(Config.TRAIN_META_PATH)
        val_meta = pd.read_csv(Config.VAL_META_PATH)
        test_meta = pd.read_csv(Config.TEST_META_PATH)

        # Load Raw Data
        # We load the full files and then index them using the IDs from metadata
        df_raw_train = pd.read_csv(Config.TRAIN_DATA_PATH).set_index("id")
        df_raw_test = pd.read_csv(Config.TEST_DATA_PATH).set_index("id")

        # Align Data using Metadata IDs
        # Metadata IDs are floats in description, but usually ints in CSV.
        # We ensure index matching.
        train_ids = train_meta["id"].values
        val_ids = val_meta["id"].values
        test_ids = test_meta["id"].values

        # Extract subsets
        # Note: df_raw_train contains both train and val samples originally
        df_train = df_raw_train.loc[train_ids].copy()
        df_val = df_raw_train.loc[val_ids].copy()
        df_test = df_raw_test.loc[test_ids].copy()

        # Feature Selection
        # f_27 is categorical, f_00-f_30 (excluding 27) are continuous
        all_cols = df_train.columns.tolist()
        cat_col = "f_27"
        target_col = "target"

        # Identify continuous columns dynamically
        cont_cols = [c for c in all_cols if c.startswith("f_") and c != cat_col]
        # Sort to ensure consistent order
        cont_cols.sort()

        if len(cont_cols) != Config.NUM_CONT_FEATURES:
            print(
                f"Warning: Expected {Config.NUM_CONT_FEATURES} continuous features, found {len(cont_cols)}"
            )

        # ----------------------------------------------------------------------
        # Preprocessing: Continuous Features (Normalization)
        # ----------------------------------------------------------------------
        print("Normalizing continuous features...")
        scaler = StandardScaler()

        # Fit ONLY on training data to prevent leakage
        train_cont = scaler.fit_transform(df_train[cont_cols].values.astype(np.float32))
        val_cont = scaler.transform(df_val[cont_cols].values.astype(np.float32))
        test_cont = scaler.transform(df_test[cont_cols].values.astype(np.float32))

        # ----------------------------------------------------------------------
        # Preprocessing: Categorical Feature (Tokenization)
        # ----------------------------------------------------------------------
        print("Tokenizing categorical feature f_27...")
        tokenizer = CharTokenizer()

        train_cat = tokenizer.transform(df_train[cat_col])
        val_cat = tokenizer.transform(df_val[cat_col])
        test_cat = tokenizer.transform(df_test[cat_col])

        # ----------------------------------------------------------------------
        # Targets
        # ----------------------------------------------------------------------
        # Use targets from metadata to be safe (ground truth source)
        train_target = train_meta[target_col].values.astype(np.float32)
        val_target = val_meta[target_col].values.astype(np.float32)

        # ----------------------------------------------------------------------
        # Save to Cache
        # ----------------------------------------------------------------------
        print(f"Saving processed data to {cache_path}...")
        np.savez_compressed(
            cache_path,
            train_cont=train_cont,
            train_cat=train_cat,
            train_target=train_target,
            val_cont=val_cont,
            val_cat=val_cat,
            val_target=val_target,
            test_cont=test_cont,
            test_cat=test_cat,
        )

    # --------------------------------------------------------------------------
    # 3. Debug Subsampling
    # --------------------------------------------------------------------------
    if debug:
        print(f"DEBUG MODE: Truncating datasets to {Config.DEBUG_SAMPLES} samples.")
        limit = Config.DEBUG_SAMPLES
        train_cont = train_cont[:limit]
        train_cat = train_cat[:limit]
        train_target = train_target[:limit]

        val_cont = val_cont[:limit]
        val_cat = val_cat[:limit]
        val_target = val_target[:limit]

        test_cont = test_cont[:limit]
        test_cat = test_cat[:limit]

    # --------------------------------------------------------------------------
    # 4. Create Datasets
    # --------------------------------------------------------------------------
    train_dataset = ManufacturingDataset(train_cont, train_cat, train_target)
    val_dataset = ManufacturingDataset(val_cont, val_cat, val_target)
    test_dataset = ManufacturingDataset(test_cont, test_cat, targets=None)

    print(
        f"Datasets prepared. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    return train_dataset, val_dataset, test_dataset
