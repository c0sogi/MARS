import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class RNATokenizer:
    """
    Tokenizes RNA sequence, structure, and predicted loop type into integer indices.
    """

    def __init__(self):
        self.seq_map = Config.TOKEN2INT_SEQ
        self.struct_map = Config.TOKEN2INT_STRUCT
        self.loop_map = Config.TOKEN2INT_LOOP

    def tokenize(self, sequence, structure, loop_type):
        """
        Converts strings to integer arrays.
        Returns a stacked array of shape (Seq_Len, 3).
        """
        # Convert strings to lists of indices
        seq_ints = [self.seq_map.get(c, 0) for c in sequence]
        struct_ints = [self.struct_map.get(c, 0) for c in structure]
        loop_ints = [self.loop_map.get(c, 0) for c in loop_type]

        # Stack into (Seq_Len, 3)
        # Channel 0: Sequence
        # Channel 1: Structure
        # Channel 2: Loop Type
        return np.stack([seq_ints, struct_ints, loop_ints], axis=1).astype(np.int64)


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA data.
    """

    def __init__(self, inputs, targets=None, ids=None):
        self.inputs = torch.from_numpy(inputs)
        self.targets = torch.from_numpy(targets) if targets is not None else None
        self.ids = ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        X = self.inputs[idx]
        if self.targets is not None:
            y = self.targets[idx]
            return X, y
        return X


def pad_targets(target_list, total_len=107):
    """
    Pads the target list (usually length 68) to total_len (107) with zeros.
    """
    arr = np.array(target_list, dtype=np.float32)
    current_len = len(arr)
    if current_len < total_len:
        padding = np.zeros(total_len - current_len, dtype=np.float32)
        arr = np.concatenate([arr, padding])
    return arr


def preprocess_and_cache(csv_path, cache_prefix, is_test=False, load_cached_data=True):
    """
    Loads data from CSV, preprocesses it, and caches it as .npy files.
    If cache exists and load_cached_data is True, loads from disk.
    """
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    inputs_path = os.path.join(cache_dir, f"{cache_prefix}_inputs.npy")
    targets_path = os.path.join(cache_dir, f"{cache_prefix}_targets.npy")
    ids_path = os.path.join(cache_dir, f"{cache_prefix}_ids.npy")

    # 1. Try to load from cache
    if load_cached_data:
        if os.path.exists(inputs_path) and os.path.exists(ids_path):
            if is_test or os.path.exists(targets_path):
                print(f"Loading {cache_prefix} data from cache...")
                inputs = np.load(inputs_path)
                ids = np.load(ids_path, allow_pickle=True)
                targets = np.load(targets_path) if not is_test else None
                return inputs, targets, ids

    # 2. Process from scratch
    print(f"Processing {cache_prefix} data from {csv_path}...")
    df = pd.read_csv(csv_path)

    tokenizer = RNATokenizer()

    # Process Inputs
    input_list = []
    for _, row in df.iterrows():
        tokenized = tokenizer.tokenize(
            row["sequence"], row["structure"], row["predicted_loop_type"]
        )
        input_list.append(tokenized)

    inputs = np.array(input_list)  # Shape: (N, 107, 3)
    ids = df["id"].values

    # Process Targets (if not test)
    targets = None
    if not is_test:
        target_list_all = []
        for _, row in df.iterrows():
            # Parse stringified lists for each target column
            # Config.TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
            row_targets = []
            for col in Config.TARGET_COLS:
                # ast.literal_eval safely evaluates the string representation of the list
                val_list = ast.literal_eval(row[col])
                padded_val = pad_targets(val_list, Config.SEQ_LEN)
                row_targets.append(padded_val)

            # Stack targets for this sample: Shape (107, 5)
            sample_targets = np.stack(row_targets, axis=1)
            target_list_all.append(sample_targets)

        targets = np.array(target_list_all)  # Shape: (N, 107, 5)

    # 3. Save to cache
    print(f"Saving {cache_prefix} data to cache...")
    np.save(inputs_path, inputs)
    np.save(ids_path, ids)
    if targets is not None:
        np.save(targets_path, targets)

    return inputs, targets, ids


def get_dataloaders(load_cached_data=True, batch_size=None, num_workers=None):
    """
    Creates DataLoaders for training and validation.
    """
    bs = batch_size if batch_size is not None else Config.BATCH_SIZE
    nw = num_workers if num_workers is not None else Config.NUM_WORKERS

    # Process Train
    train_inputs, train_targets, train_ids = preprocess_and_cache(
        Config.TRAIN_CSV, "train", is_test=False, load_cached_data=load_cached_data
    )

    # Process Val
    val_inputs, val_targets, val_ids = preprocess_and_cache(
        Config.VAL_CSV, "val", is_test=False, load_cached_data=load_cached_data
    )

    # Create Datasets
    train_dataset = RNADataset(train_inputs, train_targets, train_ids)
    val_dataset = RNADataset(val_inputs, val_targets, val_ids)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader


def get_test_dataloader(load_cached_data=True, batch_size=None, num_workers=None):
    """
    Creates DataLoader for the test set.
    """
    bs = batch_size if batch_size is not None else Config.BATCH_SIZE
    nw = num_workers if num_workers is not None else Config.NUM_WORKERS

    test_inputs, _, test_ids = preprocess_and_cache(
        Config.TEST_CSV, "test", is_test=True, load_cached_data=load_cached_data
    )

    test_dataset = RNADataset(test_inputs, targets=None, ids=test_ids)

    test_loader = DataLoader(
        test_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return test_loader, test_ids
