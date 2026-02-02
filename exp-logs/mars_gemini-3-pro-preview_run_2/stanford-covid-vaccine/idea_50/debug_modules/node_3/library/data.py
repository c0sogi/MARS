import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config


def get_structure_adj(structure, seq_len):
    """
    Parses dot-bracket structure to find pair indices.
    Returns an array of shape (seq_len,) where arr[i] is the index of the base paired with i,
    or -1 if i is unpaired.
    """
    pairs = np.full(seq_len, -1, dtype=np.int32)
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


def process_data(load_cached_data=True, debug_size=None):
    """
    Loads, processes, and caches data.
    Returns: (train_dict, val_dict, test_dict)
    Each dict contains: 'inputs', 'pair_indices', 'targets', 'ids'
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(Config.CACHE_DIR, "train_data_reid_fn_v1.npz")

    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}...")
        try:
            data = np.load(cache_file, allow_pickle=True)
            return (
                {
                    k: data[f"train_{k}"]
                    for k in ["inputs", "pair_indices", "targets", "ids"]
                },
                {
                    k: data[f"val_{k}"]
                    for k in ["inputs", "pair_indices", "targets", "ids"]
                },
                {k: data[f"test_{k}"] for k in ["inputs", "pair_indices", "ids"]},
            )
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing from scratch.")

    print("Processing data from scratch...")

    # Load Metadata
    train_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "test.csv"))

    if debug_size:
        train_df = train_df.iloc[:debug_size]
        val_df = val_df.iloc[:debug_size]
        test_df = test_df.iloc[:debug_size]

    # Helper for feature generation
    def get_features(df, is_test=False):
        # Mappings
        seq_map = {c: i for i, c in enumerate("AGCU")}
        struct_map = {c: i for i, c in enumerate(".()")}
        loop_map = {c: i for i, c in enumerate("SMIBHEX")}

        n_samples = len(df)
        seq_len = Config.SEQ_LEN

        # Channels: 4 (Seq) + 3 (Struct) + 7 (Loop) + 4 (Partner Identity) = 18
        inputs = np.zeros((n_samples, seq_len, 18), dtype=np.float32)
        pair_indices = np.full((n_samples, seq_len), -1, dtype=np.int32)

        # Targets: (N, L, 5). Initialize with zeros.
        targets = (
            np.zeros((n_samples, seq_len, 5), dtype=np.float32) if not is_test else None
        )
        ids = df["id"].values

        # Iterate over rows
        # Using enumerate(df.itertuples()) is faster than iterrows
        for i, row in enumerate(df.itertuples(index=False)):
            seq = row.sequence
            struct = row.structure
            loop = row.predicted_loop_type

            # 1. Basic One-Hot Encoding
            for j, char in enumerate(seq):
                if char in seq_map:
                    inputs[i, j, seq_map[char]] = 1.0

            for j, char in enumerate(struct):
                if char in struct_map:
                    inputs[i, j, 4 + struct_map[char]] = 1.0

            for j, char in enumerate(loop):
                if char in loop_map:
                    inputs[i, j, 7 + loop_map[char]] = 1.0

            # 2. Partner Info
            pairs = get_structure_adj(struct, seq_len)
            pair_indices[i] = pairs

            # Partner Identity Injection
            # If base j is paired with p_idx, we put the one-hot of seq[p_idx] into the input at j
            for j, p_idx in enumerate(pairs):
                if p_idx != -1:
                    partner_char = seq[p_idx]
                    if partner_char in seq_map:
                        inputs[i, j, 14 + seq_map[partner_char]] = 1.0

            # 3. Targets (Training/Validation only)
            if not is_test:
                # Parse stringified lists for each target column
                for t_idx, col in enumerate(Config.ALL_TARGET_COLS):
                    # getattr(row, col) gets the string representation of the list
                    val_str = getattr(row, col)
                    try:
                        val_list = ast.literal_eval(val_str)
                    except (ValueError, SyntaxError):
                        val_list = []

                    # Fill the target array up to the length of the provided data (usually 68)
                    len_t = len(val_list)
                    if len_t > 0:
                        targets[i, :len_t, t_idx] = val_list

        return inputs, pair_indices, targets, ids

    # Generate features
    print("Generating training features...")
    train_inputs, train_pairs, train_targets, train_ids = get_features(train_df)
    print("Generating validation features...")
    val_inputs, val_pairs, val_targets, val_ids = get_features(val_df)
    print("Generating test features...")
    test_inputs, test_pairs, _, test_ids = get_features(test_df, is_test=True)

    # Save to cache
    print(f"Saving cache to {cache_file}...")
    np.savez_compressed(
        cache_file,
        train_inputs=train_inputs,
        train_pair_indices=train_pairs,
        train_targets=train_targets,
        train_ids=train_ids,
        val_inputs=val_inputs,
        val_pair_indices=val_pairs,
        val_targets=val_targets,
        val_ids=val_ids,
        test_inputs=test_inputs,
        test_pair_indices=test_pairs,
        test_ids=test_ids,
    )

    print("Data processed and cached.")
    return (
        {
            "inputs": train_inputs,
            "pair_indices": train_pairs,
            "targets": train_targets,
            "ids": train_ids,
        },
        {
            "inputs": val_inputs,
            "pair_indices": val_pairs,
            "targets": val_targets,
            "ids": val_ids,
        },
        {"inputs": test_inputs, "pair_indices": test_pairs, "ids": test_ids},
    )


class RNADataset(Dataset):
    def __init__(self, data_dict, is_test=False):
        self.inputs = data_dict["inputs"]
        self.pair_indices = data_dict["pair_indices"]
        self.ids = data_dict["ids"]
        self.is_test = is_test
        if not is_test:
            self.targets = data_dict["targets"]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # inputs: (L, C) -> Transpose to (C, L) for PyTorch Conv1d
        x = torch.tensor(self.inputs[idx], dtype=torch.float32).transpose(0, 1)

        # pair_indices: (L,) -> LongTensor
        pairs = torch.tensor(self.pair_indices[idx], dtype=torch.long)

        if self.is_test:
            return x, pairs, self.ids[idx]
        else:
            # targets: (L, 5) -> FloatTensor
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            return x, pairs, y, self.ids[idx]
