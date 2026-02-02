import os
import hashlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed


class RNADataset(Dataset):
    def __init__(self, inputs, pair_indices, targets=None, ids=None):
        self.inputs = inputs
        self.pair_indices = pair_indices
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # inputs: (107, 14)
        sequence_data = torch.tensor(self.inputs[idx], dtype=torch.float32)

        # pair_indices: (107,)
        pair_index_data = torch.tensor(self.pair_indices[idx], dtype=torch.long)

        sample = {
            "sequence": sequence_data,
            "pair_index": pair_index_data,
            "id": self.ids[idx],
        }

        if self.targets is not None:
            # targets: (68, 5)
            target_data = torch.tensor(self.targets[idx], dtype=torch.float32)
            sample["targets"] = target_data

        return sample


def get_pair_index(structure):
    """
    Parses a dot-bracket structure string to find paired indices.
    Returns a numpy array of shape (seq_len,) where:
      arr[i] = j if i is paired with j
      arr[i] = -1 if i is unpaired
    """
    seq_len = len(structure)
    pair_index = np.full(seq_len, -1, dtype=np.int32)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                pair_index[i] = j
                pair_index[j] = i

    return pair_index


def one_hot_encode(seq, mapping, vocab_size):
    """
    Converts a sequence string into a one-hot encoded numpy array.
    """
    indices = [mapping.get(char, 0) for char in seq]
    # Create one-hot
    one_hot = np.zeros((len(seq), vocab_size), dtype=np.float32)
    one_hot[np.arange(len(seq)), indices] = 1.0
    return one_hot


def process_data(df, is_test=False):
    """
    Process the dataframe into numpy arrays for features and targets.
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LENGTH
    input_dim = Config.INPUT_DIM  # 14

    # Pre-allocate arrays
    inputs = np.zeros((num_samples, seq_len, input_dim), dtype=np.float32)
    pair_indices = np.zeros((num_samples, seq_len), dtype=np.int32)
    ids = df["id"].values

    if not is_test:
        targets = np.zeros(
            (num_samples, Config.SEQ_SCORED, Config.OUTPUT_DIM), dtype=np.float32
        )
    else:
        targets = None

    # Process each sample
    for idx, row in df.iterrows():
        # 1. Sequence Features (4 channels)
        seq_oh = one_hot_encode(
            row["sequence"], Config.TOKEN2INT_SEQ, len(Config.TOKEN2INT_SEQ)
        )

        # 2. Structure Features (3 channels)
        struct_oh = one_hot_encode(
            row["structure"], Config.TOKEN2INT_STRUCT, len(Config.TOKEN2INT_STRUCT)
        )

        # 3. Loop Type Features (7 channels)
        loop_oh = one_hot_encode(
            row["predicted_loop_type"],
            Config.TOKEN2INT_LOOP,
            len(Config.TOKEN2INT_LOOP),
        )

        # Concatenate: (107, 4+3+7=14)
        inputs[idx] = np.concatenate([seq_oh, struct_oh, loop_oh], axis=1)

        # 4. Pair Indices
        pair_indices[idx] = get_pair_index(row["structure"])

        # 5. Targets (only for train/val)
        if not is_test:
            # Targets are lists in the dataframe. We stack the 5 target columns.
            # Shape: (68, 5)
            # Columns: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
            t_list = []
            for col in Config.TARGET_COLS:
                val = row[col]
                # Ensure it's a list or array
                if not isinstance(val, (list, np.ndarray)):
                    # Fallback for safety, though parquet should handle it
                    val = [0.0] * Config.SEQ_SCORED
                t_list.append(val)

            # Transpose from (5, 68) to (68, 5)
            targets[idx] = np.array(t_list, dtype=np.float32).T

    return inputs, pair_indices, targets, ids


def get_dataset_from_cache_or_process(mode, load_cached_data=True):
    """
    Loads data from cache if available, otherwise processes from source parquet files.
    mode: 'train', 'val', or 'test'
    """
    # Determine source file
    if mode == "train":
        source_path = Config.TRAIN_FILE
    elif mode == "val":
        source_path = Config.VAL_FILE
    elif mode == "test":
        source_path = Config.TEST_FILE
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Generate a hash for caching based on mode and key config parameters
    config_str = f"{mode}_{Config.SEQ_LENGTH}_{Config.SEQ_SCORED}_{Config.INPUT_DIM}"
    config_hash = hashlib.md5(config_str.encode()).hexdigest()
    cache_path = os.path.join(Config.WORKING_DIR, f"{mode}_data_{config_hash}.npz")

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=True)
            inputs = data["inputs"]
            pair_indices = data["pair_indices"]
            ids = data["ids"]
            if mode != "test":
                targets = data["targets"]
            else:
                targets = None
            print(f"Loaded {mode} data from cache: {cache_path}")
            return RNADataset(inputs, pair_indices, targets, ids)
        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}. Reprocessing...")

    # Process from scratch
    print(f"Processing {mode} data from {source_path}...")
    df = pd.read_parquet(source_path)

    is_test = mode == "test"
    inputs, pair_indices, targets, ids = process_data(df, is_test=is_test)

    # Save to cache
    save_dict = {"inputs": inputs, "pair_indices": pair_indices, "ids": ids}
    if targets is not None:
        save_dict["targets"] = targets

    np.savez_compressed(cache_path, **save_dict)
    print(f"Saved {mode} data to cache: {cache_path}")

    return RNADataset(inputs, pair_indices, targets, ids)


def get_loaders(load_cached_data=True):
    """
    Returns DataLoaders for train, val, and test sets.
    """
    set_seed(Config.SEED)

    # Load Datasets
    train_dataset = get_dataset_from_cache_or_process("train", load_cached_data)
    val_dataset = get_dataset_from_cache_or_process("val", load_cached_data)
    test_dataset = get_dataset_from_cache_or_process("test", load_cached_data)

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
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
