import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from library.config import Config


class ManufacturingDataset(Dataset):
    """
    PyTorch Dataset for the Manufacturing Control Data.
    Implements dynamic Swap Noise corruption for the DiGUT Discriminator task.
    """

    def __init__(self, x_num, x_seq, y=None, is_train=False, config: Config = None):
        # Convert to tensors
        self.x_num = torch.FloatTensor(x_num)
        self.x_seq = torch.LongTensor(x_seq)
        self.y = torch.FloatTensor(y) if y is not None else None

        self.is_train = is_train
        self.config = config
        self.num_samples = len(x_num)

        # Dimensions for mask generation
        self.num_numerical = x_num.shape[1]
        self.seq_length = x_seq.shape[1]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1. Fetch Original Data
        x_n = self.x_num[idx]
        x_s = self.x_seq[idx]

        # Target (0.0 for test set)
        target = self.y[idx] if self.y is not None else torch.tensor(0.0)

        # 2. Apply Swap Noise (Only during training)
        # Note: Noise is now applied on GPU in trainer.py (Masking).
        # We bypass CPU-side corruption here.
        x_n_final = x_n
        x_s_final = x_s
        # Placeholder for mask/auxiliary target (unused by current trainer)
        mask_combined = torch.zeros(self.num_numerical + self.seq_length).float()

        return x_n_final, x_s_final, target, mask_combined


def preprocess_data(config: Config, load_cached_data: bool = True):
    """
    Loads raw data, performs feature engineering (unique_characters),
    standardization, and tokenization.

    Implements caching mechanism to store processed numpy arrays.

    Returns:
        Tuple containing processed training, validation, and test arrays,
        plus a metadata dictionary.
    """
    cache_dir = config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define paths for cached files
    files = {
        "X_num_train": os.path.join(cache_dir, "X_num_train.npy"),
        "X_seq_train": os.path.join(cache_dir, "X_seq_train.npy"),
        "y_train": os.path.join(cache_dir, "y_train.npy"),
        "X_num_val": os.path.join(cache_dir, "X_num_val.npy"),
        "X_seq_val": os.path.join(cache_dir, "X_seq_val.npy"),
        "y_val": os.path.join(cache_dir, "y_val.npy"),
        "X_num_test": os.path.join(cache_dir, "X_num_test.npy"),
        "X_seq_test": os.path.join(cache_dir, "X_seq_test.npy"),
        "ids_test": os.path.join(cache_dir, "ids_test.npy"),
        "meta": os.path.join(cache_dir, "meta.json"),
    }

    # Check if cache exists
    cache_valid = all(os.path.exists(p) for p in files.values())

    if load_cached_data and cache_valid:
        print(f"Loading preprocessed data from cache: {cache_dir}")
        data = {}
        for k, v in files.items():
            if k == "meta":
                with open(v, "r") as f:
                    data[k] = json.load(f)
            else:
                data[k] = np.load(v)

        return (
            data["X_num_train"],
            data["X_seq_train"],
            data["y_train"],
            data["X_num_val"],
            data["X_seq_val"],
            data["y_val"],
            data["X_num_test"],
            data["X_seq_test"],
            data["ids_test"],
            data["meta"],
        )

    print("Cache not found or disabled. Processing data from scratch...")

    # 1. Load Data
    print("Loading CSV files...")
    train_df = pd.read_csv(config.TRAIN_PATH)
    val_df = pd.read_csv(config.VAL_PATH)
    test_df = pd.read_csv(config.TEST_PATH)

    # 2. Feature Engineering: Unique Characters
    print("Engineering features...")

    def count_unique_chars(s):
        return len(set(s))

    for df in [train_df, val_df, test_df]:
        df["unique_characters"] = df["f_27"].apply(count_unique_chars)

    # 3. Identify Numerical Columns
    # Exclude non-feature columns
    exclude_cols = ["id", "target", "source_path", "f_27"]
    # Select all numerical columns (f_00...f_30 + unique_characters)
    num_cols = [c for c in train_df.columns if c not in exclude_cols]
    num_cols.sort()  # Ensure deterministic order

    print(f"Numerical features ({len(num_cols)}): {num_cols}")

    # 4. Preprocess Numerical Features (StandardScaler)
    scaler = StandardScaler()
    # Fit only on training data
    X_num_train = scaler.fit_transform(train_df[num_cols].values.astype(np.float32))
    X_num_val = scaler.transform(val_df[num_cols].values.astype(np.float32))
    X_num_test = scaler.transform(test_df[num_cols].values.astype(np.float32))

    # 5. Preprocess Sequence Feature (f_27)
    # Build vocabulary from training data
    print("Tokenizing sequences...")
    all_chars = sorted(list(set("".join(train_df["f_27"].values))))
    # 1-based indexing (0 reserved for padding/unknown if needed)
    char_to_idx = {c: i + 1 for i, c in enumerate(all_chars)}
    vocab_size = len(all_chars) + 1

    def tokenize(series):
        # Convert list of strings to numpy array of indices
        return np.array(
            [[char_to_idx.get(c, 0) for c in s] for s in series], dtype=np.int64
        )

    X_seq_train = tokenize(train_df["f_27"])
    X_seq_val = tokenize(val_df["f_27"])
    X_seq_test = tokenize(test_df["f_27"])

    # 6. Extract Targets and IDs
    y_train = train_df["target"].values.astype(np.float32)
    y_val = val_df["target"].values.astype(np.float32)
    ids_test = test_df["id"].values.astype(np.int64)

    # 7. Metadata
    meta = {
        "vocab_size": vocab_size,
        "num_numerical_features": len(num_cols),
        "sequence_length": X_seq_train.shape[1],
        "num_cols": num_cols,
    }

    # 8. Save to Cache
    print(f"Saving processed data to {cache_dir}...")
    np.save(files["X_num_train"], X_num_train)
    np.save(files["X_seq_train"], X_seq_train)
    np.save(files["y_train"], y_train)
    np.save(files["X_num_val"], X_num_val)
    np.save(files["X_seq_val"], X_seq_val)
    np.save(files["y_val"], y_val)
    np.save(files["X_num_test"], X_num_test)
    np.save(files["X_seq_test"], X_seq_test)
    np.save(files["ids_test"], ids_test)

    with open(files["meta"], "w") as f:
        json.dump(meta, f)

    print("Data processing complete.")

    return (
        X_num_train,
        X_seq_train,
        y_train,
        X_num_val,
        X_seq_val,
        y_val,
        X_num_test,
        X_seq_test,
        ids_test,
        meta,
    )
