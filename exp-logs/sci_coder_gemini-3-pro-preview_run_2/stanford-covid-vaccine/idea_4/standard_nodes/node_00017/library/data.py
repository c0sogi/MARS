import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library import config

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_couples(structure):
    """
    Parses a dot-bracket structure string to identify base pairs.
    Returns a list of (i, j) tuples where base i pairs with base j.
    """
    pairs = []
    stack = []
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                pairs.append((j, i))
                pairs.append((i, j))  # Add both directions for symmetric lookup
    return pairs


def one_hot_encode(sequence, mapping, num_categories):
    """
    Converts a sequence string into a one-hot encoded numpy array.
    """
    arr = np.zeros((len(sequence), num_categories), dtype=np.float32)
    for i, char in enumerate(sequence):
        if char in mapping:
            arr[i, mapping[char]] = 1.0
    return arr


def preprocess_inputs(df):
    """
    Generates the input tensor with shape (N, 107, 18).
    Channels: Sequence(4) + Structure(3) + Loop(7) + Partner(4).
    """
    n_samples = len(df)
    seq_len = config.SEQ_LENGTH

    # Initialize output array
    # Shape: (N, 107, 18)
    inputs = np.zeros((n_samples, seq_len, config.INPUT_CHANNELS), dtype=np.float32)

    for idx, row in df.iterrows():
        sequence = row[config.SEQUENCE_COL]
        structure = row[config.STRUCTURE_COL]
        loop_type = row[config.LOOP_TYPE_COL]

        # 1. Standard One-Hot Encodings
        seq_oh = one_hot_encode(sequence, config.TOKEN2INT_SEQ, config.NUM_SEQ_TOKENS)
        struct_oh = one_hot_encode(
            structure, config.TOKEN2INT_STRUCT, config.NUM_STRUCT_TOKENS
        )
        loop_oh = one_hot_encode(
            loop_type, config.TOKEN2INT_LOOP, config.NUM_LOOP_TOKENS
        )

        # 2. Partner Features
        # Initialize with zeros
        partner_feat = np.zeros((seq_len, config.NUM_PARTNER_TOKENS), dtype=np.float32)

        # Find pairs
        pairs = get_couples(structure)

        # Fill partner features
        # If i is paired with j, partner_feat[i] gets seq_oh[j]
        for i, j in pairs:
            if i < seq_len and j < seq_len:
                partner_feat[i] = seq_oh[j]

        # 3. Concatenate
        # Axis 1 is channels
        sample_feat = np.concatenate([seq_oh, struct_oh, loop_oh, partner_feat], axis=1)
        inputs[idx] = sample_feat

    return inputs


def preprocess_targets(df):
    """
    Parses target columns and generates target tensor with shape (N, 68, 5).
    """
    n_samples = len(df)
    seq_scored = config.SEQ_SCORED
    n_targets = len(config.TARGET_COLS)

    targets = np.zeros((n_samples, seq_scored, n_targets), dtype=np.float32)

    for idx, row in df.iterrows():
        for t_i, col_name in enumerate(config.TARGET_COLS):
            val_str = row[col_name]
            try:
                # Parse string "[0.1, 0.2, ...]" to list
                val_list = ast.literal_eval(val_str)
                val_arr = np.array(val_list, dtype=np.float32)

                # Ensure length matches seq_scored (68)
                if len(val_arr) > seq_scored:
                    val_arr = val_arr[:seq_scored]
                elif len(val_arr) < seq_scored:
                    # Pad with nans or zeros? Usually data is complete, but safety check:
                    pad = np.zeros(seq_scored - len(val_arr), dtype=np.float32)
                    val_arr = np.concatenate([val_arr, pad])

                targets[idx, :, t_i] = val_arr
            except Exception:
                # Fallback for parsing errors
                pass

    return targets


def load_or_process_data(mode, load_cached_data=True):
    """
    Loads data from cache or processes it from raw CSVs.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (ids, inputs, targets)
               ids: numpy array of strings
               inputs: numpy array (N, 107, 18)
               targets: numpy array (N, 68, 5) or None for test
    """
    # Define cache paths
    cache_dir = config.CACHE_DIR
    ids_path = os.path.join(cache_dir, f"{mode}_ids.npy")
    inputs_path = os.path.join(cache_dir, f"{mode}_inputs.npy")
    targets_path = os.path.join(cache_dir, f"{mode}_targets.npy")

    has_cache = (
        os.path.exists(ids_path)
        and os.path.exists(inputs_path)
        and (mode == "test" or os.path.exists(targets_path))
    )

    if load_cached_data and has_cache:
        # Load from cache
        ids = np.load(ids_path, allow_pickle=True)
        inputs = np.load(inputs_path)
        if mode != "test":
            targets = np.load(targets_path)
        else:
            targets = None
        return ids, inputs, targets

    # Process from scratch
    if mode == "train":
        csv_path = config.TRAIN_CSV
    elif mode == "val":
        csv_path = config.VAL_CSV
    else:
        csv_path = config.TEST_CSV

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Debugging subset
    if config.DEBUG:
        df = df.head(config.DEBUG_SUBSET_SIZE)

    ids = df[config.ID_COL].values
    inputs = preprocess_inputs(df)

    if mode != "test":
        targets = preprocess_targets(df)
    else:
        targets = None

    # Save to cache
    np.save(ids_path, ids)
    np.save(inputs_path, inputs)
    if mode != "test":
        np.save(targets_path, targets)

    return ids, inputs, targets


# =============================================================================
# DATASET CLASS
# =============================================================================


class RNADataset(Dataset):
    def __init__(self, inputs, targets=None, ids=None):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.ids = ids
        if targets is not None:
            self.targets = torch.tensor(targets, dtype=torch.float32)
        else:
            self.targets = None

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if self.targets is not None:
            return self.inputs[idx], self.targets[idx]
        else:
            return self.inputs[idx]


# =============================================================================
# DATA LOADERS
# =============================================================================


def get_dataloaders(load_cached_data=True):
    """
    Main function to get DataLoaders for train, val, and test.
    """
    # Load Data
    train_ids, train_inputs, train_targets = load_or_process_data(
        "train", load_cached_data
    )
    val_ids, val_inputs, val_targets = load_or_process_data("val", load_cached_data)
    test_ids, test_inputs, _ = load_or_process_data("test", load_cached_data)

    # Create Datasets
    train_dataset = RNADataset(train_inputs, train_targets, train_ids)
    val_dataset = RNADataset(val_inputs, val_targets, val_ids)
    test_dataset = RNADataset(test_inputs, targets=None, ids=test_ids)

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader
