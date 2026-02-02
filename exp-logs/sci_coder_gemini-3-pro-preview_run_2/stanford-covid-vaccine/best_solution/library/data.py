import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything

# Dictionaries for One-Hot Encoding
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {"(": 0, ".": 1, ")": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def get_partner_indices(structure):
    """
    Parses dot-bracket structure to find base pairs.
    Returns an array where arr[i] is the index of the partner of base i,
    or -1 if unpaired.
    """
    partner_indices = np.full(len(structure), -1, dtype=np.int32)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                partner_indices[i] = j
                partner_indices[j] = i

    return partner_indices


def one_hot_encode(sequence, mapping, num_classes):
    """
    Converts a sequence string into a one-hot encoded numpy array.
    """
    seq_len = len(sequence)
    encoding = np.zeros((seq_len, num_classes), dtype=np.float32)
    for i, char in enumerate(sequence):
        if char in mapping:
            encoding[i, mapping[char]] = 1.0
    return encoding


def process_dataframe(df, mode="train"):
    """
    Processes the dataframe to generate features and targets.
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LENGTH

    # Pre-allocate arrays
    # Channels: 4 (Seq) + 3 (Struct) + 7 (Loop) + 4 (PartnerID) = 18
    input_features = np.zeros(
        (num_samples, seq_len, Config.INPUT_CHANNELS), dtype=np.float32
    )
    partner_indices_all = np.zeros((num_samples, seq_len), dtype=np.int32)

    # Targets: 5 channels (reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C)
    # We pad to seq_len (107) even though only first 68 are scored.
    targets = np.zeros((num_samples, seq_len, Config.NUM_TARGETS), dtype=np.float32)

    ids = df["id"].values

    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for idx, row in df.iterrows():
        # 1. Base Features
        seq_oh = one_hot_encode(row["sequence"], SEQ_MAP, 4)
        struct_oh = one_hot_encode(row["structure"], STRUCT_MAP, 3)
        loop_oh = one_hot_encode(row["predicted_loop_type"], LOOP_MAP, 7)

        # 2. Partner Indices
        p_indices = get_partner_indices(row["structure"])
        partner_indices_all[idx] = p_indices

        # 3. Partner Identity
        # If i is paired with j, partner_id[i] = one_hot(sequence[j])
        # If i is unpaired, partner_id[i] = 0
        partner_id_oh = np.zeros((seq_len, 4), dtype=np.float32)
        for i, partner_idx in enumerate(p_indices):
            if partner_idx != -1:
                partner_id_oh[i] = seq_oh[partner_idx]

        # Concatenate all features
        # Shape: (107, 18)
        sample_features = np.concatenate(
            [seq_oh, struct_oh, loop_oh, partner_id_oh], axis=1
        )
        input_features[idx] = sample_features

        # 4. Targets (only for train/val)
        if mode in ["train", "val"]:
            for t_i, col in enumerate(target_cols):
                # Targets are stored as stringified lists in the CSV
                try:
                    val_list = ast.literal_eval(row[col])
                    # Fill the valid length (usually 68)
                    valid_len = min(len(val_list), seq_len)
                    targets[idx, :valid_len, t_i] = val_list[:valid_len]
                except (ValueError, SyntaxError):
                    pass  # Keep zeros if parsing fails or empty

    return input_features, partner_indices_all, targets, ids


def load_or_process_data(csv_path, cache_path, mode="train", load_cached_data=True):
    """
    Loads data from cache if available, otherwise processes from CSV and caches it.
    """
    # Ensure cache directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            data = np.load(cache_path, allow_pickle=True)
            return data["inputs"], data["partner_indices"], data["targets"], data["ids"]
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    print(f"Processing data from {csv_path}...")
    df = pd.read_csv(csv_path)

    inputs, partner_indices, targets, ids = process_dataframe(df, mode=mode)

    print(f"Saving processed data to {cache_path}...")
    np.savez_compressed(
        cache_path,
        inputs=inputs,
        partner_indices=partner_indices,
        targets=targets,
        ids=ids,
    )

    return inputs, partner_indices, targets, ids


class RNADataset(Dataset):
    def __init__(self, inputs, partner_indices, targets, ids, mode="train"):
        self.inputs = inputs
        self.partner_indices = partner_indices
        self.targets = targets
        self.ids = ids
        self.mode = mode

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # inputs: (Seq, Channels)
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)

        # partner_indices: (Seq,)
        p_idx = torch.tensor(self.partner_indices[idx], dtype=torch.long)

        # targets: (Seq, 5)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)

        # In test mode, targets are dummy zeros, which is fine
        return x, p_idx, y


def get_loader(
    mode="train", batch_size=None, num_workers=None, load_cached_data=True, debug=False
):
    """
    Creates and returns a DataLoader for the specified mode.
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    if num_workers is None:
        num_workers = Config.NUM_WORKERS

    if mode == "train":
        csv_path = Config.TRAIN_METADATA
        cache_path = Config.TRAIN_CACHE
        shuffle = True
    elif mode == "val":
        csv_path = Config.VAL_METADATA
        cache_path = Config.VAL_CACHE
        shuffle = False
    elif mode == "test":
        csv_path = Config.TEST_METADATA
        cache_path = Config.TEST_CACHE
        shuffle = False
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Load data
    inputs, partner_indices, targets, ids = load_or_process_data(
        csv_path, cache_path, mode=mode, load_cached_data=load_cached_data
    )

    # Debug subsampling
    if debug:
        debug_size = min(len(inputs), Config.DEBUG_SIZE)
        inputs = inputs[:debug_size]
        partner_indices = partner_indices[:debug_size]
        targets = targets[:debug_size]
        ids = ids[:debug_size]
        print(f"DEBUG MODE: Subsampled dataset to {debug_size} samples.")

    dataset = RNADataset(inputs, partner_indices, targets, ids, mode=mode)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    return loader
