import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed

# Dictionaries for One-Hot Encoding
SEQ_MAP = {"A": [1, 0, 0, 0], "G": [0, 1, 0, 0], "C": [0, 0, 1, 0], "U": [0, 0, 0, 1]}

STRUCT_MAP = {"(": [1, 0, 0], ")": [0, 1, 0], ".": [0, 0, 1]}

LOOP_MAP = {
    "S": [1, 0, 0, 0, 0, 0, 0],
    "M": [0, 1, 0, 0, 0, 0, 0],
    "I": [0, 0, 1, 0, 0, 0, 0],
    "B": [0, 0, 0, 1, 0, 0, 0],
    "H": [0, 0, 0, 0, 1, 0, 0],
    "E": [0, 0, 0, 0, 0, 1, 0],
    "X": [0, 0, 0, 0, 0, 0, 1],
}


def get_partner_map(structure):
    """
    Parses dot-bracket structure to find base pairs.
    Returns an array of indices where arr[i] is the index of the base paired with i.
    Returns -1 if unpaired.
    """
    partner_map = np.full(len(structure), -1, dtype=np.int32)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                partner_map[i] = j
                partner_map[j] = i

    return partner_map


def get_partner_identity(sequence_one_hot, partner_map):
    """
    Constructs the partner identity features.
    For each position i, if it has a partner j, this feature is the one-hot encoding of base j.
    If unpaired, it is a zero vector.
    """
    seq_len, num_bases = sequence_one_hot.shape
    partner_identity = np.zeros((seq_len, num_bases), dtype=np.float32)

    for i in range(seq_len):
        partner_idx = partner_map[i]
        if partner_idx != -1:
            partner_identity[i] = sequence_one_hot[partner_idx]

    return partner_identity


def process_sequence(sequence, structure, loop_type):
    """
    Generates the full input feature tensor for a single sample.
    Concatenates: Sequence(4) + Structure(3) + LoopType(7) + PartnerIdentity(4).
    Returns:
        features: (Seq_Len, 18)
        partner_map: (Seq_Len,)
    """
    # 1. Basic One-Hot Encodings
    seq_oh = np.array(
        [SEQ_MAP.get(c, [0, 0, 0, 0]) for c in sequence], dtype=np.float32
    )
    struct_oh = np.array(
        [STRUCT_MAP.get(c, [0, 0, 1]) for c in structure], dtype=np.float32
    )
    loop_oh = np.array(
        [LOOP_MAP.get(c, [0, 0, 0, 0, 0, 0, 1]) for c in loop_type], dtype=np.float32
    )

    # 2. Structural Context
    partner_map = get_partner_map(structure)
    partner_identity = get_partner_identity(seq_oh, partner_map)

    # 3. Concatenate
    # Shapes: (L, 4), (L, 3), (L, 7), (L, 4) -> (L, 18)
    features = np.concatenate([seq_oh, struct_oh, loop_oh, partner_identity], axis=1)

    return features, partner_map


class RNADataset(Dataset):
    def __init__(self, inputs, partner_indices, targets=None, ids=None):
        """
        Args:
            inputs (np.ndarray): Shape (N, Seq_Len, Input_Channels)
            partner_indices (np.ndarray): Shape (N, Seq_Len)
            targets (np.ndarray, optional): Shape (N, Seq_Len, Num_Targets)
            ids (list, optional): List of sample IDs
        """
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.partner_indices = torch.tensor(partner_indices, dtype=torch.long)
        self.ids = ids

        if targets is not None:
            self.targets = torch.tensor(targets, dtype=torch.float32)
        else:
            self.targets = None

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Returns: Input, Partner_Map, Target (if available)
        X = self.inputs[idx]
        P = self.partner_indices[idx]

        if self.targets is not None:
            Y = self.targets[idx]
            return X, P, Y
        else:
            return X, P


def load_data(mode="train", load_cached_data=True):
    """
    Loads data for the specified mode ('train', 'val', 'test').
    Handles caching to .npz files to speed up subsequent runs.

    Args:
        mode (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        DataLoader: A PyTorch DataLoader for the requested dataset.
    """
    set_seed()

    # Determine file paths based on mode
    if mode == "train":
        csv_path = os.path.join(Config.METADATA_DIR, "train.csv")
        cache_file = Config.TRAIN_CACHE_FILE
        is_test = False
    elif mode == "val":
        csv_path = os.path.join(Config.METADATA_DIR, "val.csv")
        cache_file = Config.VAL_CACHE_FILE
        is_test = False
    elif mode == "test":
        csv_path = os.path.join(Config.METADATA_DIR, "test.csv")
        cache_file = Config.TEST_CACHE_FILE
        is_test = True
    else:
        raise ValueError(f"Invalid mode: {mode}")

    cache_path = Config.get_cache_path(cache_file)

    # 1. Try Loading from Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {mode} data from cache: {cache_path}")
        try:
            data = np.load(cache_path, allow_pickle=True)
            inputs = data["inputs"]
            partner_indices = data["partner_indices"]
            if is_test:
                ids = data["ids"]
                targets = None
            else:
                targets = data["targets"]
                ids = data["ids"]  # Optional, but good to have

            # Create Dataset and Loader
            dataset = RNADataset(inputs, partner_indices, targets, ids)
            shuffle = mode == "train"
            return DataLoader(
                dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=shuffle,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing data...")

    # 2. Process from Scratch
    print(f"Processing {mode} data from CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # Debug subset
    if Config.SUBSET_SIZE is not None:
        df = df.head(Config.SUBSET_SIZE)

    inputs_list = []
    partner_indices_list = []
    targets_list = []
    ids_list = df["id"].tolist()

    # Pre-parse target columns if training/val
    if not is_test:
        for col in Config.TARGET_COLS:
            df[col] = df[col].apply(
                lambda x: np.array(ast.literal_eval(x), dtype=np.float32)
            )

    for idx, row in df.iterrows():
        # Feature Engineering
        feats, p_map = process_sequence(
            row["sequence"], row["structure"], row["predicted_loop_type"]
        )
        inputs_list.append(feats)
        partner_indices_list.append(p_map)

        # Target Processing (Anchoring)
        if not is_test:
            # Initialize target array of shape (107, 5) with zeros (Anchoring)
            target_matrix = np.zeros(
                (Config.SEQ_LEN, Config.NUM_TARGETS), dtype=np.float32
            )

            # Fill the first 68 positions with ground truth
            for t_i, col in enumerate(Config.TARGET_COLS):
                raw_vals = row[col]
                # Safety check for length
                length = min(len(raw_vals), Config.SCORED_LEN)
                target_matrix[:length, t_i] = raw_vals[:length]

            targets_list.append(target_matrix)

    # Convert to numpy arrays
    inputs_arr = np.array(inputs_list, dtype=np.float32)
    partner_indices_arr = np.array(partner_indices_list, dtype=np.int32)

    if not is_test:
        targets_arr = np.array(targets_list, dtype=np.float32)
    else:
        targets_arr = None

    # 3. Save to Cache
    print(f"Saving {mode} data to cache: {cache_path}")
    save_dict = {
        "inputs": inputs_arr,
        "partner_indices": partner_indices_arr,
        "ids": ids_list,
    }
    if not is_test:
        save_dict["targets"] = targets_arr

    np.savez_compressed(cache_path, **save_dict)

    # 4. Return DataLoader
    dataset = RNADataset(inputs_arr, partner_indices_arr, targets_arr, ids_list)
    shuffle = mode == "train"

    return DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
