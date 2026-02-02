import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.utils import seed_everything


class Tokenizer:
    """
    Handles tokenization of character sequences.
    Maps characters to integers and handles padding.
    """

    def __init__(self, max_len=10):
        self.char_to_idx = {}
        self.vocab_size = 0
        self.max_len = max_len

    def fit(self, series):
        """
        Builds vocabulary from a pandas Series of strings.
        """
        all_chars = set()
        for s in series:
            all_chars.update(s)

        # Sort for deterministic vocabulary
        vocab = sorted(list(all_chars))
        # 0 is reserved for padding/masking
        self.char_to_idx = {c: i + 1 for i, c in enumerate(vocab)}
        self.vocab_size = len(vocab) + 1
        return self

    def transform(self, series):
        """
        Converts a Series of strings to a numpy array of token indices.
        """
        seqs = []
        for s in series:
            seq = [self.char_to_idx.get(c, 0) for c in s]
            # Pad or truncate to max_len
            if len(seq) < self.max_len:
                seq += [0] * (self.max_len - len(seq))
            else:
                seq = seq[: self.max_len]
            seqs.append(seq)
        return np.array(seqs, dtype=np.int64)


class FeatureEngineer:
    """
    Handles feature engineering tasks.
    """

    def transform(self, df):
        """
        Adds engineered features to the dataframe.
        """
        df = df.copy()
        # Extract unique character count from f_27
        df["unique_characters"] = df["f_27"].apply(lambda x: len(set(x)))
        return df


class ManufacturingDataset(Dataset):
    """
    PyTorch Dataset for the manufacturing data.
    Implements stochastic masking for the transformer branch.
    """

    def __init__(self, X_num, X_seq, y=None, mask_prob=0.0):
        self.X_num = X_num
        self.X_seq = X_seq
        self.y = y
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.X_num)

    def __getitem__(self, idx):
        x_n = self.X_num[idx]
        x_s = self.X_seq[idx]

        # Logic for masking
        if self.mask_prob > 0:
            # Sequence Masking
            prob_s = np.random.rand(len(x_s))
            mask_indices_s = prob_s < self.mask_prob
            x_s_masked = x_s.copy()
            x_s_masked[mask_indices_s] = 0  # 0 is mask token
            mask_s = mask_indices_s.astype(bool)
            x_s_out = x_s_masked

            # Numerical Masking
            prob_n = np.random.rand(len(x_n))
            mask_indices_n = prob_n < self.mask_prob
            mask_num = mask_indices_n.astype(bool)
        else:
            x_s_out = x_s
            mask_s = np.zeros_like(x_s, dtype=bool)
            mask_num = np.zeros_like(x_n, dtype=bool)

        # Target sequence for reconstruction is the original sequence
        target_s = x_s

        # Convert to tensors
        item = {
            "x_num": torch.tensor(x_n, dtype=torch.float32),
            "mask_num": torch.tensor(mask_num, dtype=torch.bool),
            "x_seq": torch.tensor(x_s_out, dtype=torch.long),
            "target_seq": torch.tensor(target_s, dtype=torch.long),
            "mask_seq": torch.tensor(mask_s, dtype=torch.bool),
        }

        if self.y is not None:
            item["target"] = torch.tensor(self.y[idx], dtype=torch.float32)

        return item


def process_data(load_cached_data=True):
    """
    Orchestrates data loading, feature engineering, tokenization, scaling, and caching.

    Args:
        load_cached_data (bool): If True, attempts to load data from .npy files.

    Returns:
        data (dict): Dictionary containing processed numpy arrays.
        vocab_size (int): Size of the character vocabulary.
    """
    seed_everything(Config.SEED)

    # Ensure working directory exists
    os.makedirs(Config.WORK_DIR, exist_ok=True)

    # Define cache file paths
    cache_files = {
        "X_num_train": os.path.join(Config.WORK_DIR, "X_num_train.npy"),
        "X_seq_train": os.path.join(Config.WORK_DIR, "X_seq_train.npy"),
        "y_train": os.path.join(Config.WORK_DIR, "y_train.npy"),
        "X_num_val": os.path.join(Config.WORK_DIR, "X_num_val.npy"),
        "X_seq_val": os.path.join(Config.WORK_DIR, "X_seq_val.npy"),
        "y_val": os.path.join(Config.WORK_DIR, "y_val.npy"),
        "X_num_test": os.path.join(Config.WORK_DIR, "X_num_test.npy"),
        "X_seq_test": os.path.join(Config.WORK_DIR, "X_seq_test.npy"),
        "ids_test": os.path.join(Config.WORK_DIR, "ids_test.npy"),
        "vocab_size": os.path.join(Config.WORK_DIR, "vocab_size.npy"),
    }

    # Check if all cache files exist
    cache_exists = all(os.path.exists(p) for p in cache_files.values())

    if load_cached_data and cache_exists:
        print(f"Loading cached data from {Config.WORK_DIR}...")
        data = {}
        for k, v in cache_files.items():
            if k != "vocab_size":
                data[k] = np.load(v)

        vocab_size = int(np.load(cache_files["vocab_size"]))
        return data, vocab_size

    print("Processing data from scratch...")

    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_META)
    val_df = pd.read_csv(Config.VAL_META)
    test_df = pd.read_csv(Config.TEST_META)

    # Feature Engineering
    fe = FeatureEngineer()
    train_df = fe.transform(train_df)
    val_df = fe.transform(val_df)
    test_df = fe.transform(test_df)

    # Tokenization
    tokenizer = Tokenizer()
    tokenizer.fit(train_df["f_27"])

    X_seq_train = tokenizer.transform(train_df["f_27"])
    X_seq_val = tokenizer.transform(val_df["f_27"])
    X_seq_test = tokenizer.transform(test_df["f_27"])
    vocab_size = tokenizer.vocab_size

    # Numerical Processing
    # Identify numerical columns: All columns except id, target, f_27, source_path
    exclude_cols = ["id", "target", "f_27", "source_path"]
    num_cols = [c for c in train_df.columns if c not in exclude_cols]

    scaler = StandardScaler()
    X_num_train = scaler.fit_transform(train_df[num_cols].values.astype(np.float32))
    X_num_val = scaler.transform(val_df[num_cols].values.astype(np.float32))
    X_num_test = scaler.transform(test_df[num_cols].values.astype(np.float32))

    # Targets and IDs
    y_train = train_df["target"].values.astype(np.float32)
    y_val = val_df["target"].values.astype(np.float32)
    ids_test = test_df["id"].values

    # Save to Cache
    np.save(cache_files["X_num_train"], X_num_train)
    np.save(cache_files["X_seq_train"], X_seq_train)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["X_num_val"], X_num_val)
    np.save(cache_files["X_seq_val"], X_seq_val)
    np.save(cache_files["y_val"], y_val)
    np.save(cache_files["X_num_test"], X_num_test)
    np.save(cache_files["X_seq_test"], X_seq_test)
    np.save(cache_files["ids_test"], ids_test)
    np.save(cache_files["vocab_size"], np.array([vocab_size]))

    data = {
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

    return data, vocab_size
