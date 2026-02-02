import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config

# =========================================================================
# Mappings
# =========================================================================
# Sequence: 4 bases
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}

# Structure: 3 states
STRUCT_MAP = {"(": 0, ")": 1, ".": 2}

# Loop Type: 7 types
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def get_structure_adj(structure_str):
    """
    Parses a dot-bracket structure string to create an adjacency map.
    Returns a numpy array of shape (L,) where index i contains the index of the
    base paired with i, or -1 if unpaired.
    """
    length = len(structure_str)
    adj_map = np.full(length, -1, dtype=np.int64)
    stack = []

    for i, char in enumerate(structure_str):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                adj_map[i] = j
                adj_map[j] = i

    return adj_map


def one_hot_encode(sequence, structure, loop_type):
    """
    Creates the input feature tensor of shape (L, 14).
    Channels:
    0-3: Sequence (A, G, C, U)
    4-6: Structure ((, ), .)
    7-13: Loop Type (S, M, I, B, H, E, X)
    """
    length = len(sequence)
    # 4 + 3 + 7 = 14 features
    encoding = np.zeros((length, 14), dtype=np.float32)

    for i in range(length):
        # Sequence
        s_char = sequence[i]
        if s_char in SEQ_MAP:
            encoding[i, SEQ_MAP[s_char]] = 1.0

        # Structure
        st_char = structure[i]
        if st_char in STRUCT_MAP:
            encoding[i, 4 + STRUCT_MAP[st_char]] = 1.0

        # Loop Type
        l_char = loop_type[i]
        if l_char in LOOP_MAP:
            encoding[i, 7 + LOOP_MAP[l_char]] = 1.0

    return encoding


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA Degradation Prediction.
    """

    def __init__(self, inputs, adj_maps, targets=None, ids=None):
        """
        Args:
            inputs: (N, 107, 14) numpy array
            adj_maps: (N, 107) numpy array
            targets: (N, 68, 5) numpy array (optional)
            ids: List or array of sample IDs (optional)
        """
        self.inputs = inputs
        self.adj_maps = adj_maps
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Convert to tensors
        x = torch.from_numpy(self.inputs[idx]).float()
        adj = torch.from_numpy(self.adj_maps[idx]).long()

        sample = {"inputs": x, "adj_map": adj}

        if self.targets is not None:
            y = torch.from_numpy(self.targets[idx]).float()
            sample["targets"] = y

        if self.ids is not None:
            sample["id"] = self.ids[idx]

        return sample


def process_data(df, is_test=False):
    """
    Internal function to process a dataframe into numpy arrays.
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LEN
    num_features = Config.NUM_FEATURES

    # Pre-allocate arrays
    inputs = np.zeros((num_samples, seq_len, num_features), dtype=np.float32)
    adj_maps = np.zeros((num_samples, seq_len), dtype=np.int64)
    ids = df["id"].values

    # Process inputs
    sequences = df["sequence"].values
    structures = df["structure"].values
    loop_types = df["predicted_loop_type"].values

    for i in range(num_samples):
        inputs[i] = one_hot_encode(sequences[i], structures[i], loop_types[i])
        adj_maps[i] = get_structure_adj(structures[i])

    targets = None
    if not is_test:
        # Process targets
        # Targets are lists in the dataframe columns. We need to stack them.
        # Columns: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
        target_cols = Config.TARGET_COLS

        # We assume all target lists have length equal to seq_scored (68)
        seq_scored = Config.SEQ_SCORED
        targets = np.zeros(
            (num_samples, seq_scored, len(target_cols)), dtype=np.float32
        )

        for idx, col in enumerate(target_cols):
            # Convert column of lists to 2D numpy array
            # Parquet preserves lists, so we can just stack them
            col_data = np.vstack(df[col].values)
            targets[:, :, idx] = col_data

    return inputs, adj_maps, targets, ids


def get_train_val_datasets(load_cached_data=True, debug=False):
    """
    Loads training and validation data, handling caching and preprocessing.

    Args:
        load_cached_data (bool): If True, attempts to load from .npz cache.
        debug (bool): If True, uses a small subset of data.

    Returns:
        train_dataset (RNADataset), val_dataset (RNADataset)
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache paths
    train_cache = Config.TRAIN_CACHE
    val_cache = Config.VAL_CACHE

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(train_cache) and os.path.exists(val_cache):
        try:
            train_data = np.load(train_cache, allow_pickle=True)
            val_data = np.load(val_cache, allow_pickle=True)

            # Unpack
            train_inputs = train_data["inputs"]
            train_adj = train_data["adj_maps"]
            train_targets = train_data["targets"]
            train_ids = train_data["ids"]

            val_inputs = val_data["inputs"]
            val_adj = val_data["adj_maps"]
            val_targets = val_data["targets"]
            val_ids = val_data["ids"]

            # Handle debug mode on cached data
            if debug:
                subset = Config.DEBUG_SUBSET_SIZE
                train_inputs = train_inputs[:subset]
                train_adj = train_adj[:subset]
                train_targets = train_targets[:subset]
                train_ids = train_ids[:subset]

                val_inputs = val_inputs[:subset]
                val_adj = val_adj[:subset]
                val_targets = val_targets[:subset]
                val_ids = val_ids[:subset]

            return RNADataset(
                train_inputs, train_adj, train_targets, train_ids
            ), RNADataset(val_inputs, val_adj, val_targets, val_ids)

        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing data...")

    # 2. Process from scratch
    # Load metadata
    train_df = pd.read_parquet(Config.TRAIN_METADATA)
    val_df = pd.read_parquet(Config.VAL_METADATA)

    if debug:
        train_df = train_df.head(Config.DEBUG_SUBSET_SIZE)
        val_df = val_df.head(Config.DEBUG_SUBSET_SIZE)

    # Process
    train_inputs, train_adj, train_targets, train_ids = process_data(
        train_df, is_test=False
    )
    val_inputs, val_adj, val_targets, val_ids = process_data(val_df, is_test=False)

    # 3. Save to cache (only if not debugging, to keep cache full)
    if not debug:
        np.savez(
            train_cache,
            inputs=train_inputs,
            adj_maps=train_adj,
            targets=train_targets,
            ids=train_ids,
        )
        np.savez(
            val_cache,
            inputs=val_inputs,
            adj_maps=val_adj,
            targets=val_targets,
            ids=val_ids,
        )

    return RNADataset(train_inputs, train_adj, train_targets, train_ids), RNADataset(
        val_inputs, val_adj, val_targets, val_ids
    )


def get_test_dataset(load_cached_data=True, debug=False):
    """
    Loads test data, handling caching and preprocessing.

    Args:
        load_cached_data (bool): If True, attempts to load from .npz cache.
        debug (bool): If True, uses a small subset of data.

    Returns:
        test_dataset (RNADataset)
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    test_cache = Config.TEST_CACHE

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(test_cache):
        try:
            test_data = np.load(test_cache, allow_pickle=True)
            inputs = test_data["inputs"]
            adj_maps = test_data["adj_maps"]
            ids = test_data["ids"]

            if debug:
                subset = Config.DEBUG_SUBSET_SIZE
                inputs = inputs[:subset]
                adj_maps = adj_maps[:subset]
                ids = ids[:subset]

            return RNADataset(inputs, adj_maps, targets=None, ids=ids)
        except Exception as e:
            print(f"Failed to load test cache: {e}. Reprocessing data...")

    # 2. Process from scratch
    test_df = pd.read_parquet(Config.TEST_METADATA)

    if debug:
        test_df = test_df.head(Config.DEBUG_SUBSET_SIZE)

    inputs, adj_maps, _, ids = process_data(test_df, is_test=True)

    # 3. Save to cache
    if not debug:
        np.savez(test_cache, inputs=inputs, adj_maps=adj_maps, ids=ids)

    return RNADataset(inputs, adj_maps, targets=None, ids=ids)
