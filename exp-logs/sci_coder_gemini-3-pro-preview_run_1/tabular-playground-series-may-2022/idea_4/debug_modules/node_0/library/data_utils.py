import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config


# Ensure reproducibility
def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CharTokenizer:
    """
    Simple character-level tokenizer for the f_27 sequence feature.
    Maps characters to integers. 0 is reserved for padding/unknown.
    """

    def __init__(self):
        self.vocab = {}
        self.inv_vocab = {}

    def fit(self, texts):
        """Builds vocabulary from a list of strings."""
        unique_chars = set()
        for text in texts:
            unique_chars.update(list(str(text)))

        # Sort characters to ensure deterministic mapping
        sorted_chars = sorted(list(unique_chars))

        # Start indices from 1 (0 is reserved for padding)
        self.vocab = {char: idx + 1 for idx, char in enumerate(sorted_chars)}
        self.inv_vocab = {idx: char for char, idx in self.vocab.items()}

    def transform(self, texts, max_len):
        """Converts strings to integer sequences."""
        seqs = []
        for text in texts:
            text_str = str(text)
            # Truncate if necessary
            chars = list(text_str)[:max_len]
            # Map chars to indices, use 0 for unknown
            seq = [self.vocab.get(c, 0) for c in chars]
            # Pad if necessary
            if len(seq) < max_len:
                seq += [0] * (max_len - len(seq))
            seqs.append(seq)
        return np.array(seqs, dtype=np.int64)

    def get_vocab_size(self):
        # +1 for padding index 0
        return len(self.vocab) + 1


class ManufacturingDataset(Dataset):
    """
    PyTorch Dataset for the manufacturing task.
    Serves tokenized sequences and normalized numerical features.
    """

    def __init__(self, sequences, numerical, targets=None):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.numerical = torch.tensor(numerical, dtype=torch.float32)
        self.targets = (
            torch.tensor(targets, dtype=torch.float32) if targets is not None else None
        )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.targets is not None:
            return self.sequences[idx], self.numerical[idx], self.targets[idx]
        else:
            return self.sequences[idx], self.numerical[idx]


def get_dataloaders():
    """
    Main function to prepare data.
    Handles caching, preprocessing, and DataLoader creation.
    """
    set_seeds(Config.SEED)

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache file paths
    cache_files = {
        "X_seq_train": os.path.join(Config.WORKING_DIR, "X_seq_train.npy"),
        "X_num_train": os.path.join(Config.WORKING_DIR, "X_num_train.npy"),
        "y_train": os.path.join(Config.WORKING_DIR, "y_train.npy"),
        "X_seq_val": os.path.join(Config.WORKING_DIR, "X_seq_val.npy"),
        "X_num_val": os.path.join(Config.WORKING_DIR, "X_num_val.npy"),
        "y_val": os.path.join(Config.WORKING_DIR, "y_val.npy"),
        "X_seq_test": os.path.join(Config.WORKING_DIR, "X_seq_test.npy"),
        "X_num_test": os.path.join(Config.WORKING_DIR, "X_num_test.npy"),
        "vocab": os.path.join(Config.WORKING_DIR, "vocab.json"),
    }

    # Check if we can load from cache
    files_exist = all(os.path.exists(p) for p in cache_files.values())
    load_cache = Config.LOAD_CACHED_DATA and files_exist

    vocab_map = {}

    if load_cache:
        print("Loading processed data from cache...")
        X_seq_train = np.load(cache_files["X_seq_train"])
        X_num_train = np.load(cache_files["X_num_train"])
        y_train = np.load(cache_files["y_train"])

        X_seq_val = np.load(cache_files["X_seq_val"])
        X_num_val = np.load(cache_files["X_num_val"])
        y_val = np.load(cache_files["y_val"])

        X_seq_test = np.load(cache_files["X_seq_test"])
        X_num_test = np.load(cache_files["X_num_test"])

        with open(cache_files["vocab"], "r") as f:
            vocab_map = json.load(f)

    else:
        print("Processing data from scratch...")
        # Load metadata CSVs
        df_train = pd.read_csv(Config.TRAIN_PATH)
        df_val = pd.read_csv(Config.VAL_PATH)
        df_test = pd.read_csv(Config.TEST_PATH)

        # 1. Process Sequences (f_27)
        tokenizer = CharTokenizer()
        tokenizer.fit(df_train[Config.SEQ_FEATURE].astype(str).tolist())
        vocab_map = tokenizer.vocab

        X_seq_train = tokenizer.transform(
            df_train[Config.SEQ_FEATURE].astype(str).tolist(), Config.MAX_SEQ_LEN
        )
        X_seq_val = tokenizer.transform(
            df_val[Config.SEQ_FEATURE].astype(str).tolist(), Config.MAX_SEQ_LEN
        )
        X_seq_test = tokenizer.transform(
            df_test[Config.SEQ_FEATURE].astype(str).tolist(), Config.MAX_SEQ_LEN
        )

        # 2. Process Numerical Features
        scaler = StandardScaler()
        # Fit only on train
        scaler.fit(df_train[Config.NUM_FEATURES])

        X_num_train = scaler.transform(df_train[Config.NUM_FEATURES]).astype(np.float32)
        X_num_val = scaler.transform(df_val[Config.NUM_FEATURES]).astype(np.float32)
        X_num_test = scaler.transform(df_test[Config.NUM_FEATURES]).astype(np.float32)

        # 3. Targets
        y_train = df_train[Config.TARGET_COL].values.astype(np.float32)
        y_val = df_val[Config.TARGET_COL].values.astype(np.float32)

        # Save to cache
        np.save(cache_files["X_seq_train"], X_seq_train)
        np.save(cache_files["X_num_train"], X_num_train)
        np.save(cache_files["y_train"], y_train)

        np.save(cache_files["X_seq_val"], X_seq_val)
        np.save(cache_files["X_num_val"], X_num_val)
        np.save(cache_files["y_val"], y_val)

        np.save(cache_files["X_seq_test"], X_seq_test)
        np.save(cache_files["X_num_test"], X_num_test)

        with open(cache_files["vocab"], "w") as f:
            json.dump(vocab_map, f)

    # Update Config VOCAB_SIZE dynamically based on actual data
    # +1 because 0 is used for padding
    Config.VOCAB_SIZE = len(vocab_map) + 1
    print(f"Vocabulary Size: {Config.VOCAB_SIZE}")

    # Handle Debugging
    if Config.DEBUG_SAMPLE_SIZE is not None:
        print(f"Debug Mode: Slicing datasets to {Config.DEBUG_SAMPLE_SIZE} samples.")
        limit = Config.DEBUG_SAMPLE_SIZE
        X_seq_train = X_seq_train[:limit]
        X_num_train = X_num_train[:limit]
        y_train = y_train[:limit]

        X_seq_val = X_seq_val[:limit]
        X_num_val = X_num_val[:limit]
        y_val = y_val[:limit]

        X_seq_test = X_seq_test[:limit]
        X_num_test = X_num_test[:limit]

    # Create Datasets
    train_dataset = ManufacturingDataset(X_seq_train, X_num_train, y_train)
    val_dataset = ManufacturingDataset(X_seq_val, X_num_val, y_val)
    test_dataset = ManufacturingDataset(X_seq_test, X_num_test, targets=None)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
