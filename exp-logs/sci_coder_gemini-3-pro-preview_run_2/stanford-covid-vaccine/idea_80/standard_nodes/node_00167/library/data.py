import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA Degradation Prediction.
    Serves inputs, partner indices, and targets.
    """

    def __init__(self, inputs, partner_indices, targets=None, ids=None):
        self.inputs = inputs
        self.partner_indices = partner_indices
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # inputs: (Seq_Len, Input_Channels)
        # partner_indices: (Seq_Len,)
        item = {
            "inputs": torch.tensor(self.inputs[idx], dtype=torch.float32),
            "partner_indices": torch.tensor(
                self.partner_indices[idx], dtype=torch.long
            ),
        }

        # targets: (Seq_Len, 5) - Full length with zero padding in tail
        if self.targets is not None:
            item["targets"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        if self.ids is not None:
            item["id"] = self.ids[idx]

        return item


def parse_structure_pairs(structure):
    """
    Parses a dot-bracket structure string to find base pairs.
    Returns an array where arr[i] = j if i pairs with j.
    If i is unpaired, arr[i] = i (self-loop).
    """
    L = len(structure)
    pairs = np.arange(L)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                pairs[i] = j
                pairs[j] = i
    return pairs


def get_one_hot(seq, mapping):
    """
    Converts a sequence string into a One-Hot encoded numpy array.
    """
    L = len(seq)
    K = len(mapping)
    char_to_int = {c: i for i, c in enumerate(mapping)}

    one_hot = np.zeros((L, K), dtype=np.float32)
    for i, char in enumerate(seq):
        if char in char_to_int:
            one_hot[i, char_to_int[char]] = 1.0
    return one_hot


def process_dataframe(df, mode="train"):
    """
    Processes a pandas DataFrame into numpy arrays suitable for the AS-DRN model.
    Handles feature extraction, partner mapping, and target padding.
    """
    # Feature Mappings
    seq_map = ["A", "G", "C", "U"]
    struct_map = ["(", ")", "."]
    loop_map = ["S", "M", "I", "B", "H", "E", "X"]

    num_samples = len(df)
    seq_len = Config.SEQ_LENGTH

    # Calculate input channel dimension
    # Seq(4) + Struct(3) + Loop(7) + PartnerID(4) = 18 channels
    input_dim = 4 + 3 + 7 + 4

    # Initialize arrays
    inputs = np.zeros((num_samples, seq_len, input_dim), dtype=np.float32)
    partner_indices = np.zeros((num_samples, seq_len), dtype=np.int32)

    # Targets logic
    has_targets = mode in ["train", "val"]
    if has_targets:
        targets = np.zeros((num_samples, seq_len, Config.NUM_TARGETS), dtype=np.float32)
    else:
        targets = None

    ids = df["id"].values
    target_cols = Config.TARGET_COLS

    for idx, row in df.iterrows():
        # --- Input Processing ---
        sequence = row["sequence"]
        structure = row["structure"]
        loop_type = row["predicted_loop_type"]

        # 1. Basic One-Hot Encodings
        oh_seq = get_one_hot(sequence, seq_map)  # (L, 4)
        oh_struct = get_one_hot(structure, struct_map)  # (L, 3)
        oh_loop = get_one_hot(loop_type, loop_map)  # (L, 7)

        # 2. Partner Logic
        pairs = parse_structure_pairs(structure)
        partner_indices[idx] = pairs

        # Partner Identity Feature: The sequence identity of the paired base
        oh_partner = np.zeros((seq_len, 4), dtype=np.float32)
        for i in range(seq_len):
            j = pairs[i]
            if i != j:  # If paired
                oh_partner[i] = oh_seq[j]

        # Concatenate all features
        inputs[idx] = np.concatenate([oh_seq, oh_struct, oh_loop, oh_partner], axis=1)

        # --- Target Processing (Boundary Anchoring) ---
        if has_targets:
            for t_i, col in enumerate(target_cols):
                val_str = row[col]
                try:
                    # Parse stringified list from CSV
                    val_list = ast.literal_eval(val_str)
                    val_arr = np.array(val_list, dtype=np.float32)

                    # Fill valid positions (0-67)
                    # Positions 68-107 remain 0.0 (neutral baseline for anchoring)
                    length = min(len(val_arr), Config.SCORED_LENGTH)
                    targets[idx, :length, t_i] = val_arr[:length]
                except Exception:
                    # Fallback for malformed data (should not happen in clean metadata)
                    pass

    return inputs, partner_indices, targets, ids


def get_data(load_cached_data=True):
    """
    Retrieves data for train, val, and test sets.
    Implements caching mechanism using .npz files in the working directory.
    """
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file paths with versioning key
    cache_file_train = os.path.join(cache_dir, "train_data_as_drn_v1.npz")
    cache_file_val = os.path.join(cache_dir, "val_data_as_drn_v1.npz")
    cache_file_test = os.path.join(cache_dir, "test_data_as_drn_v1.npz")

    # Check if all cache files exist
    if (
        load_cached_data
        and os.path.exists(cache_file_train)
        and os.path.exists(cache_file_val)
        and os.path.exists(cache_file_test)
    ):

        print(f"Loading cached data from {cache_dir}...")
        train_data = np.load(cache_file_train, allow_pickle=True)
        val_data = np.load(cache_file_val, allow_pickle=True)
        test_data = np.load(cache_file_test, allow_pickle=True)

        return {
            "train": (
                train_data["inputs"],
                train_data["partner_indices"],
                train_data["targets"],
                train_data["ids"],
            ),
            "val": (
                val_data["inputs"],
                val_data["partner_indices"],
                val_data["targets"],
                val_data["ids"],
            ),
            "test": (
                test_data["inputs"],
                test_data["partner_indices"],
                test_data["ids"],
            ),
        }

    print("Cache not found or disabled. Processing data from scratch...")

    # Load Metadata CSVs
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)
    df_test = pd.read_csv(Config.TEST_CSV)

    # Process Dataframes
    train_inputs, train_partners, train_targets, train_ids = process_dataframe(
        df_train, mode="train"
    )
    val_inputs, val_partners, val_targets, val_ids = process_dataframe(
        df_val, mode="val"
    )
    test_inputs, test_partners, _, test_ids = process_dataframe(df_test, mode="test")

    # Save to Cache
    print(f"Saving processed data to {cache_dir}...")
    np.savez(
        cache_file_train,
        inputs=train_inputs,
        partner_indices=train_partners,
        targets=train_targets,
        ids=train_ids,
    )
    np.savez(
        cache_file_val,
        inputs=val_inputs,
        partner_indices=val_partners,
        targets=val_targets,
        ids=val_ids,
    )
    np.savez(
        cache_file_test, inputs=test_inputs, partner_indices=test_partners, ids=test_ids
    )

    return {
        "train": (train_inputs, train_partners, train_targets, train_ids),
        "val": (val_inputs, val_partners, val_targets, val_ids),
        "test": (test_inputs, test_partners, test_ids),
    }


def get_dataloaders(debug=False):
    """
    Creates and returns PyTorch DataLoaders for train, val, and test sets.

    Args:
        debug (bool): If True, subsets the data to Config.DEBUG_SAMPLES for rapid testing.
    """
    # Load data (cached or fresh)
    data = get_data(load_cached_data=True)

    train_inputs, train_partners, train_targets, train_ids = data["train"]
    val_inputs, val_partners, val_targets, val_ids = data["val"]
    test_inputs, test_partners, test_ids = data["test"]

    # Apply Debug Subsetting
    if debug:
        limit = Config.DEBUG_SAMPLES
        print(f"Debug mode enabled: Limiting data to {limit} samples.")
        train_inputs = train_inputs[:limit]
        train_partners = train_partners[:limit]
        train_targets = train_targets[:limit]
        train_ids = train_ids[:limit]

        val_inputs = val_inputs[:limit]
        val_partners = val_partners[:limit]
        val_targets = val_targets[:limit]
        val_ids = val_ids[:limit]
        # We generally want full test set even in debug to ensure submission code works,
        # but for speed we can limit it too if strictly debugging pipeline.
        test_inputs = test_inputs[:limit]
        test_partners = test_partners[:limit]
        test_ids = test_ids[:limit]

    # Create Datasets
    train_dataset = RNADataset(train_inputs, train_partners, train_targets, train_ids)
    val_dataset = RNADataset(val_inputs, val_partners, val_targets, val_ids)
    test_dataset = RNADataset(test_inputs, test_partners, None, test_ids)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
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
