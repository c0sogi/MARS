import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import parse_structure_to_pairs

# One-hot encoding mappings
SEQ_MAP = {c: i for i, c in enumerate("AGCU")}
STRUCT_MAP = {c: i for i, c in enumerate("().")}
LOOP_MAP = {c: i for i, c in enumerate("SMIBHEX")}


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA degradation prediction.
    Returns inputs, topology information (pair indices/masks), and targets.
    """

    def __init__(self, inputs, pair_indices, pair_masks, targets=None, ids=None):
        self.inputs = inputs
        self.pair_indices = pair_indices
        self.pair_masks = pair_masks
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Convert numpy arrays to tensors
        item = {
            "inputs": torch.tensor(self.inputs[idx], dtype=torch.float32),
            "pair_indices": torch.tensor(self.pair_indices[idx], dtype=torch.long),
            "pair_mask": torch.tensor(self.pair_masks[idx], dtype=torch.float32),
        }

        if self.targets is not None:
            item["targets"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        if self.ids is not None:
            item["ids"] = self.ids[idx]

        return item


def one_hot_encode(seq, mapping, length):
    """
    Helper to one-hot encode a string sequence.
    """
    arr = np.zeros((length, len(mapping)), dtype=np.float32)
    for i, char in enumerate(seq):
        if i >= length:
            break
        if char in mapping:
            arr[i, mapping[char]] = 1.0
    return arr


def process_data(df, mode):
    """
    Processes a pandas DataFrame into numpy arrays for the model.
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LEN
    input_dim = Config.INPUT_DIM  # 14

    # Pre-allocate arrays
    inputs = np.zeros((num_samples, seq_len, input_dim), dtype=np.float32)
    pair_indices = np.zeros((num_samples, seq_len), dtype=np.int64)
    pair_masks = np.zeros((num_samples, seq_len), dtype=np.float32)
    ids = df["id"].values

    # Targets are only present in train and val sets
    targets = None
    if mode in ["train", "val"]:
        # 5 target columns, length 107 (padded)
        targets = np.zeros((num_samples, seq_len, Config.NUM_TARGETS), dtype=np.float32)

    for idx, row in df.iterrows():
        # 1. Feature Encoding
        seq_enc = one_hot_encode(row["sequence"], SEQ_MAP, seq_len)
        struct_enc = one_hot_encode(row["structure"], STRUCT_MAP, seq_len)
        loop_enc = one_hot_encode(row["predicted_loop_type"], LOOP_MAP, seq_len)

        # Concatenate features: (L, 4) + (L, 3) + (L, 7) -> (L, 14)
        inputs[idx] = np.concatenate([seq_enc, struct_enc, loop_enc], axis=1)

        # 2. Topology Parsing
        # parse_structure_to_pairs returns indices and mask for the specific structure length
        p_idx, p_mask = parse_structure_to_pairs(row["structure"])

        # Handle length consistency (though data should be 107)
        curr_len = len(p_idx)
        if curr_len > seq_len:
            pair_indices[idx] = p_idx[:seq_len]
            pair_masks[idx] = p_mask[:seq_len]
        else:
            pair_indices[idx, :curr_len] = p_idx
            pair_masks[idx, :curr_len] = p_mask
            # Fill remaining with self-loops (index pointing to itself)
            if curr_len < seq_len:
                pair_indices[idx, curr_len:] = np.arange(curr_len, seq_len)

        # 3. Target Processing
        if targets is not None:
            for t_i, col in enumerate(Config.TARGET_COLS):
                val_list = row[col]
                # Targets are provided as lists (length 68 in this dataset)
                # We pad the rest with zeros (masked out during loss calculation)
                if isinstance(val_list, (list, np.ndarray)):
                    valid_len = min(len(val_list), seq_len)
                    targets[idx, :valid_len, t_i] = val_list[:valid_len]

    return inputs, pair_indices, pair_masks, targets, ids


def get_data_arrays(mode, load_cached_data=True):
    """
    Retrieves data arrays, using caching to speed up subsequent runs.
    """
    cache_file = os.path.join(Config.CACHE_DIR, f"{mode}_data.npz")

    # Attempt to load from cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading {mode} data from cache: {cache_file}")
        try:
            data = np.load(cache_file, allow_pickle=True)
            inputs = data["inputs"]
            pair_indices = data["pair_indices"]
            pair_masks = data["pair_masks"]
            ids = data["ids"]

            # Handle targets
            targets = None
            if "targets" in data:
                # npz might store None as a 0-d array object if saved explicitly,
                # but usually we only save the key if it exists.
                if data["targets"].shape != ():
                    targets = data["targets"]

            return inputs, pair_indices, pair_masks, targets, ids
        except Exception as e:
            print(f"Cache load failed ({e}). Recomputing...")

    # Compute from scratch
    print(f"Processing {mode} data from metadata...")
    if mode == "train":
        path = Config.TRAIN_METADATA_PATH
    elif mode == "val":
        path = Config.VAL_METADATA_PATH
    else:
        path = Config.TEST_METADATA_PATH

    df = pd.read_parquet(path)
    inputs, pair_indices, pair_masks, targets, ids = process_data(df, mode)

    # Save to cache
    print(f"Saving {mode} data to cache: {cache_file}")
    save_dict = {
        "inputs": inputs,
        "pair_indices": pair_indices,
        "pair_masks": pair_masks,
        "ids": ids,
    }
    if targets is not None:
        save_dict["targets"] = targets

    np.savez_compressed(cache_file, **save_dict)

    return inputs, pair_indices, pair_masks, targets, ids


def get_dataloaders(
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
    debug=False,
):
    """
    Generates DataLoaders for train, validation, and test sets.
    """
    # Load all data arrays
    train_data = get_data_arrays("train", load_cached_data)
    val_data = get_data_arrays("val", load_cached_data)
    test_data = get_data_arrays("test", load_cached_data)

    # Unpack
    train_inputs, train_pairs, train_masks, train_targets, train_ids = train_data
    val_inputs, val_pairs, val_masks, val_targets, val_ids = val_data
    test_inputs, test_pairs, test_masks, test_targets, test_ids = test_data

    # Handle Debug Mode (Subset data)
    if debug:
        subset_size = Config.DEBUG_SUBSET_SIZE
        print(f"DEBUG MODE: Reducing dataset sizes to {subset_size}")

        train_inputs = train_inputs[:subset_size]
        train_pairs = train_pairs[:subset_size]
        train_masks = train_masks[:subset_size]
        train_targets = train_targets[:subset_size]
        train_ids = train_ids[:subset_size]

        val_inputs = val_inputs[:subset_size]
        val_pairs = val_pairs[:subset_size]
        val_masks = val_masks[:subset_size]
        val_targets = val_targets[:subset_size]
        val_ids = val_ids[:subset_size]

        test_inputs = test_inputs[:subset_size]
        test_pairs = test_pairs[:subset_size]
        test_masks = test_masks[:subset_size]
        test_ids = test_ids[:subset_size]

    # Initialize Datasets
    train_dataset = RNADataset(
        train_inputs, train_pairs, train_masks, train_targets, train_ids
    )
    val_dataset = RNADataset(val_inputs, val_pairs, val_masks, val_targets, val_ids)
    test_dataset = RNADataset(test_inputs, test_pairs, test_masks, None, test_ids)

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
