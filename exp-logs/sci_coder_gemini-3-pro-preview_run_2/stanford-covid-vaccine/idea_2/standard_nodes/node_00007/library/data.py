import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# =============================================================================
# Mappings
# =============================================================================
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

# =============================================================================
# Helper Functions
# =============================================================================


def structure_to_edge_index(structure):
    """
    Parses a dot-bracket structure string to generate graph connectivity.

    Args:
        structure (str): Dot-bracket notation string (e.g., ".(..).").

    Returns:
        np.ndarray: Edge indices of shape (2, E), where E is the number of edges.
                    Includes both backbone connections and hydrogen bonds.
    """
    seq_len = len(structure)
    src = []
    dst = []

    # 1. Backbone connections (linear chain)
    for i in range(seq_len - 1):
        # Forward
        src.append(i)
        dst.append(i + 1)
        # Backward
        src.append(i + 1)
        dst.append(i)

    # 2. Hydrogen bond connections (base pairs)
    stack = []
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                # Undirected edge i <-> j
                src.append(i)
                dst.append(j)
                src.append(j)
                dst.append(i)

    if len(src) == 0:
        return np.zeros((2, 0), dtype=np.int64)

    return np.array([src, dst], dtype=np.int64)


def get_one_hot(seq, mapping):
    """
    Converts a sequence string into a one-hot encoded numpy array.
    """
    arr = np.zeros((len(seq), len(mapping)), dtype=np.float32)
    for i, char in enumerate(seq):
        if char in mapping:
            arr[i, mapping[char]] = 1.0
    return arr


def parse_target_column(col_str, seq_len=107):
    """
    Parses a stringified list from the CSV and pads it to seq_len.
    """
    try:
        # Parse string "[0.1, 0.2, ...]" to list
        val_list = ast.literal_eval(col_str)
        arr = np.array(val_list, dtype=np.float32)

        # Pad with zeros if shorter than seq_len
        if len(arr) < seq_len:
            pad_width = seq_len - len(arr)
            arr = np.pad(arr, (0, pad_width), mode="constant", constant_values=0)
        elif len(arr) > seq_len:
            arr = arr[:seq_len]

        return arr
    except Exception:
        # Fallback for errors or NaNs
        return np.zeros(seq_len, dtype=np.float32)


# =============================================================================
# Dataset Class
# =============================================================================


class RNADataset(Dataset):
    def __init__(self, inputs, adj, ids, targets=None):
        """
        Args:
            inputs (np.ndarray): Feature matrix (N, Seq_Len, Channels).
            adj (np.ndarray): Adjacency matrix (N, Seq_Len, Seq_Len).
            ids (np.ndarray): Array of sample IDs.
            targets (np.ndarray, optional): Target matrix (N, Seq_Len, Num_Targets).
        """
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.adj = torch.tensor(adj, dtype=torch.float32)
        self.ids = ids

        if targets is not None:
            self.targets = torch.tensor(targets, dtype=torch.float32)
        else:
            self.targets = None

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        sample = {
            "inputs": self.inputs[idx],  # (Seq_Len, Channels)
            "adj": self.adj[idx],  # (Seq_Len, Seq_Len)
            "id": self.ids[idx],
        }

        if self.targets is not None:
            sample["targets"] = self.targets[idx]  # (Seq_Len, Num_Targets)

        return sample


# =============================================================================
# Processing and Loading Logic
# =============================================================================


def process_dataframe(df, config, is_test=False):
    """
    Process a dataframe into numpy arrays for inputs, adjacency, and targets.
    """
    num_samples = len(df)
    seq_len = config.seq_len

    # Initialize arrays
    # Channels: 4 (Seq) + 3 (Struct) + 7 (Loop) = 14
    inputs = np.zeros((num_samples, seq_len, config.input_channels), dtype=np.float32)
    adj = np.zeros((num_samples, seq_len, seq_len), dtype=np.float32)
    ids = df["id"].values

    targets = None
    if not is_test:
        targets = np.zeros((num_samples, seq_len, config.num_targets), dtype=np.float32)

    for idx, row in df.iterrows():
        # 1. Input Features
        # Sequence One-Hot
        seq_oh = get_one_hot(row["sequence"], SEQ_MAP)
        # Structure One-Hot
        struct_oh = get_one_hot(row["structure"], STRUCT_MAP)
        # Loop Type One-Hot
        loop_oh = get_one_hot(row["predicted_loop_type"], LOOP_MAP)

        # Concatenate features
        # Shape: (107, 14)
        inputs[idx] = np.concatenate([seq_oh, struct_oh, loop_oh], axis=1)

        # 2. Adjacency Matrix
        edge_index = structure_to_edge_index(row["structure"])
        if edge_index.shape[1] > 0:
            # edge_index is (2, E), use it to fill adjacency
            adj[idx, edge_index[0], edge_index[1]] = 1.0

        # 3. Targets (if not test)
        if not is_test:
            for t_i, col in enumerate(config.target_cols):
                targets[idx, :, t_i] = parse_target_column(row[col], seq_len)

    return inputs, adj, targets, ids


def get_dataloaders(config, load_cached_data=True):
    """
    Main function to get train, val, and test dataloaders.
    Handles caching of processed numpy arrays.
    """
    # Ensure working directory exists
    os.makedirs(config.working_dir, exist_ok=True)

    splits = ["train", "val", "test"]
    data_cache = {}

    for split in splits:
        # Define cache filenames
        cache_inputs = os.path.join(config.working_dir, f"{split}_inputs.npy")
        cache_adj = os.path.join(config.working_dir, f"{split}_adj.npy")
        cache_targets = os.path.join(config.working_dir, f"{split}_targets.npy")
        cache_ids = os.path.join(config.working_dir, f"{split}_ids.npy")

        # Determine if we need to process from scratch
        files_exist = (
            os.path.exists(cache_inputs)
            and os.path.exists(cache_adj)
            and os.path.exists(cache_ids)
            and (split == "test" or os.path.exists(cache_targets))
        )

        should_load = load_cached_data and files_exist

        if should_load:
            print(f"Loading cached {split} data...")
            inputs = np.load(cache_inputs)
            adj = np.load(cache_adj)
            ids = np.load(cache_ids, allow_pickle=True)
            targets = np.load(cache_targets) if split != "test" else None
        else:
            print(f"Processing {split} data from metadata...")
            # Load CSV
            if split == "train":
                csv_path = config.train_file
            elif split == "val":
                csv_path = config.val_file
            else:
                csv_path = config.test_file

            df = pd.read_csv(csv_path)

            # Debug subset
            if config.debug:
                df = df.head(config.subset_size)

            # Process
            inputs, adj, targets, ids = process_dataframe(
                df, config, is_test=(split == "test")
            )

            # Save to cache
            np.save(cache_inputs, inputs)
            np.save(cache_adj, adj)
            np.save(cache_ids, ids)
            if targets is not None:
                np.save(cache_targets, targets)

        data_cache[split] = (inputs, adj, targets, ids)

    # Create Datasets
    # Explicitly unpack to handle argument order mismatch (Cite debug_lesson_2)
    # process_dataframe returns: (inputs, adj, targets, ids)
    # RNADataset expects: (inputs, adj, ids, targets)

    train_inputs, train_adj, train_targets, train_ids = data_cache["train"]
    train_ds = RNADataset(train_inputs, train_adj, train_ids, train_targets)

    val_inputs, val_adj, val_targets, val_ids = data_cache["val"]
    val_ds = RNADataset(val_inputs, val_adj, val_ids, val_targets)

    test_inputs, test_adj, test_targets, test_ids = data_cache["test"]
    test_ds = RNADataset(test_inputs, test_adj, test_ids, test_targets)

    # Create Dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
