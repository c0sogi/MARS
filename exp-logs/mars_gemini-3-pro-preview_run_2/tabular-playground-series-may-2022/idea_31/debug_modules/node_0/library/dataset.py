import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from library.config import Config


class ManufacturingDataset(Dataset):
    def __init__(self, seq, cont, target=None):
        self.seq = torch.tensor(seq, dtype=torch.long)
        self.cont = torch.tensor(cont, dtype=torch.float32)
        self.target = (
            torch.tensor(target, dtype=torch.float32) if target is not None else None
        )

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        if self.target is not None:
            return self.seq[idx], self.cont[idx], self.target[idx]
        else:
            return self.seq[idx], self.cont[idx]


def process_data(config, load_cached_data=True):
    """
    Loads, processes, and caches the manufacturing dataset.

    Args:
        config: Configuration class containing paths and params.
        load_cached_data (bool): If True, attempts to load from disk first.

    Returns:
        Tuple containing training, validation, and test arrays.
    """
    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    cache_path = config.CACHE_PATH

    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        try:
            # allow_pickle=True is required to load the dictionary structure of npz
            # The actual data arrays are numeric
            data = np.load(cache_path, allow_pickle=True)
            return (
                data["X_train_seq"],
                data["X_train_cont"],
                data["y_train"],
                data["X_val_seq"],
                data["X_val_cont"],
                data["y_val"],
                data["X_test_seq"],
                data["X_test_cont"],
                data["test_ids"],
            )
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing from scratch...")

    print("Processing data from scratch...")

    # 2. Load Metadata
    train_meta = pd.read_csv(os.path.join(config.METADATA_DIR, "train_metadata.csv"))
    val_meta = pd.read_csv(os.path.join(config.METADATA_DIR, "val_metadata.csv"))
    test_meta = pd.read_csv(os.path.join(config.METADATA_DIR, "test_metadata.csv"))

    # 3. Load Raw Data
    # Reading full CSVs once is more efficient than reading per sample
    df_train_raw = pd.read_csv(os.path.join(config.INPUT_DIR, "train.csv"))
    df_test_raw = pd.read_csv(os.path.join(config.INPUT_DIR, "test.csv"))

    # 4. Feature Extraction Helper
    def extract_features(meta_df, raw_df, is_test=False):
        # Merge metadata with raw data to get features for specific IDs
        df = meta_df.merge(raw_df, on="id", how="left")

        # Sequence Feature (f_27)
        # Map A=1, B=2, ..., Z=26. 0 is pad/unknown.
        # We assume fixed length of 10 based on EDA/Config
        seq_len = config.SEQ_LEN

        # Fill NaNs just in case, though dataset is clean
        f27_series = df["f_27"].fillna("A" * seq_len).astype(str)

        # Vectorized-style list comprehension for string processing
        def encode_string(s):
            # Truncate to max length
            s = s[:seq_len]
            # Convert chars to 1-26 ints
            encoded = [max(0, min(26, ord(c) - ord("A") + 1)) for c in s]
            # Pad if shorter than seq_len
            if len(encoded) < seq_len:
                encoded += [0] * (seq_len - len(encoded))
            return encoded

        # Create sequence matrix
        seq_list = [encode_string(s) for s in f27_series]
        seq_data = np.array(seq_list, dtype=np.int32)

        # Continuous Features
        # Columns f_00 to f_30, excluding f_27
        cont_cols = [f"f_{i:02d}" for i in range(31) if i != 27]
        cont_data = df[cont_cols].values.astype(np.float32)

        if is_test:
            return seq_data, cont_data, df["id"].values
        else:
            return seq_data, cont_data, df["target"].values

    # Extract Features
    print("Extracting training features...")
    X_train_seq, X_train_cont, y_train = extract_features(train_meta, df_train_raw)

    print("Extracting validation features...")
    X_val_seq, X_val_cont, y_val = extract_features(val_meta, df_train_raw)

    print("Extracting test features...")
    X_test_seq, X_test_cont, test_ids = extract_features(
        test_meta, df_test_raw, is_test=True
    )

    # 5. Normalization
    print("Normalizing continuous features...")
    scaler = StandardScaler()

    # Fit ONLY on training data to avoid leakage
    X_train_cont = scaler.fit_transform(X_train_cont)

    # Transform validation and test
    X_val_cont = scaler.transform(X_val_cont)
    X_test_cont = scaler.transform(X_test_cont)

    # 6. Cache Data
    print(f"Saving processed data to {cache_path}...")
    np.savez(
        cache_path,
        X_train_seq=X_train_seq,
        X_train_cont=X_train_cont,
        y_train=y_train,
        X_val_seq=X_val_seq,
        X_val_cont=X_val_cont,
        y_val=y_val,
        X_test_seq=X_test_seq,
        X_test_cont=X_test_cont,
        test_ids=test_ids,
    )

    return (
        X_train_seq,
        X_train_cont,
        y_train,
        X_val_seq,
        X_val_cont,
        y_val,
        X_test_seq,
        X_test_cont,
        test_ids,
    )
