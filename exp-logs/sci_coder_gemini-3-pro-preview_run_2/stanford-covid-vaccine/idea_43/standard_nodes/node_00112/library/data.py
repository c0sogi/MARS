import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# Mapping dictionaries for one-hot encoding
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {".": 0, "(": 1, ")": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def get_structure_pairs(structure):
    """
    Parses dot-bracket structure to find paired indices.
    Returns an array of size len(structure) where arr[i] is the index of the partner,
    or -1 if unpaired.
    """
    pairs = np.full(len(structure), -1, dtype=np.int32)
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


def one_hot_encode(seq, mapping, vocab_size):
    """
    One-hot encodes a sequence string based on a mapping dictionary.
    """
    arr = np.zeros((len(seq), vocab_size), dtype=np.float32)
    for i, char in enumerate(seq):
        if char in mapping:
            arr[i, mapping[char]] = 1.0
    return arr


def get_partner_identity(sequence, pairs):
    """
    Generates one-hot encoding of the partner base.
    If unpaired (pairs[i] == -1), returns a zero vector.
    """
    seq_len = len(sequence)
    # 4 channels for A, G, C, U
    partner_encoding = np.zeros((seq_len, 4), dtype=np.float32)

    seq_indices = [SEQ_MAP.get(c, -1) for c in sequence]

    for i in range(seq_len):
        partner_idx = pairs[i]
        if partner_idx != -1:
            base_idx = seq_indices[partner_idx]
            if base_idx != -1:
                partner_encoding[i, base_idx] = 1.0

    return partner_encoding


class RNADataset(Dataset):
    def __init__(self, inputs, partner_indices, targets=None, masks=None):
        self.inputs = inputs
        self.partner_indices = partner_indices
        self.targets = targets
        self.masks = masks

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Inputs: (Seq_Len, Channels)
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)

        # Partner Indices: (Seq_Len,)
        p_idx = torch.tensor(self.partner_indices[idx], dtype=torch.long)

        sample = {"inputs": x, "partner_indices": p_idx}

        if self.targets is not None:
            # Targets: (Seq_Len, 5)
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            sample["targets"] = y

        if self.masks is not None:
            # Mask: (Seq_Len,)
            m = torch.tensor(self.masks[idx], dtype=torch.bool)
            sample["mask"] = m

        return sample


def process_dataframe(df, config, mode="train"):
    """
    Processes a dataframe into numpy arrays for inputs, partner indices, targets, and masks.
    """
    num_samples = len(df)
    seq_len = config.seq_len

    # Initialize arrays
    # Channels: Seq(4) + Struct(3) + Loop(7) + PartnerID(4) = 18
    input_channels = 4 + 3 + 7 + 4
    inputs = np.zeros((num_samples, seq_len, input_channels), dtype=np.float32)
    partner_indices = np.zeros((num_samples, seq_len), dtype=np.int32)
    masks = np.zeros((num_samples, seq_len), dtype=np.bool_)

    # Targets: 5 channels
    targets = (
        np.zeros((num_samples, seq_len, 5), dtype=np.float32)
        if mode != "test"
        else None
    )

    # Target columns to parse
    target_cols = (
        config.target_cols
    )  # ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

    for i, row in df.iterrows():
        # 1. Sequence Features
        seq = row["sequence"]
        seq_oh = one_hot_encode(seq, SEQ_MAP, 4)

        # 2. Structure Features
        struct = row["structure"]
        struct_oh = one_hot_encode(struct, STRUCT_MAP, 3)

        # 3. Loop Type Features
        loop = row["predicted_loop_type"]
        loop_oh = one_hot_encode(loop, LOOP_MAP, 7)

        # 4. Partner Identity & Indices
        pairs = get_structure_pairs(struct)
        partner_oh = get_partner_identity(seq, pairs)

        # Concatenate inputs
        # Shape: (Seq_Len, 18)
        inputs[i] = np.concatenate([seq_oh, struct_oh, loop_oh, partner_oh], axis=1)
        partner_indices[i] = pairs

        # 5. Mask
        # seq_scored denotes how many positions have valid ground truth
        scored_len = row["seq_scored"]
        masks[i, :scored_len] = True

        # 6. Targets (Train/Val only)
        if mode != "test":
            for t_idx, col_name in enumerate(target_cols):
                val_str = row[col_name]
                try:
                    # Parse stringified list
                    val_list = ast.literal_eval(val_str)
                    # Convert to array
                    val_arr = np.array(val_list, dtype=np.float32)

                    # Handle length mismatch: targets are usually length 68, sequence is 107
                    # We fill the first len(val_arr) positions
                    curr_len = len(val_arr)
                    targets[i, :curr_len, t_idx] = val_arr
                except Exception:
                    # If parsing fails or data is missing, leave as zeros (masked out anyway)
                    pass

    return inputs, partner_indices, targets, masks


def load_or_process_data(
    csv_path, cache_path, config, mode="train", load_cached_data=True
):
    """
    Loads data from cache if available and requested, otherwise processes from CSV and caches.
    """
    # Ensure cache directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        data = np.load(cache_path)
        inputs = data["inputs"]
        partner_indices = data["partner_indices"]
        masks = data["masks"]
        if mode != "test":
            targets = data["targets"]
            return inputs, partner_indices, targets, masks
        else:
            return inputs, partner_indices, None, masks

    print(f"Processing data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Debug mode: subset data
    if config.debug:
        df = df.iloc[: config.batch_size * 2]

    inputs, partner_indices, targets, masks = process_dataframe(df, config, mode)

    print(f"Saving data to {cache_path}")
    if mode != "test":
        np.savez_compressed(
            cache_path,
            inputs=inputs,
            partner_indices=partner_indices,
            targets=targets,
            masks=masks,
        )
    else:
        np.savez_compressed(
            cache_path, inputs=inputs, partner_indices=partner_indices, masks=masks
        )

    return inputs, partner_indices, targets, masks


def get_loaders(config, load_cached_data=True):
    """
    Generates DataLoaders for train, validation, and test sets.
    """
    # Load Train Data
    train_inputs, train_p_idx, train_targets, train_masks = load_or_process_data(
        config.train_csv,
        config.train_cache,
        config,
        mode="train",
        load_cached_data=load_cached_data,
    )

    # Load Val Data
    val_inputs, val_p_idx, val_targets, val_masks = load_or_process_data(
        config.val_csv,
        config.val_cache,
        config,
        mode="val",
        load_cached_data=load_cached_data,
    )

    # Load Test Data
    test_inputs, test_p_idx, _, test_masks = load_or_process_data(
        config.test_csv,
        config.test_cache,
        config,
        mode="test",
        load_cached_data=load_cached_data,
    )

    # Create Datasets
    train_dataset = RNADataset(train_inputs, train_p_idx, train_targets, train_masks)
    val_dataset = RNADataset(val_inputs, val_p_idx, val_targets, val_masks)
    test_dataset = RNADataset(test_inputs, test_p_idx, None, test_masks)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
