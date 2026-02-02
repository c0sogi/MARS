import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# ==========================================
# Mappings
# ==========================================
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {".": 0, "(": 1, ")": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def get_partner_map(structure):
    """
    Parses dot-bracket structure to find paired indices.
    Returns an array where arr[i] is the index of the partner of i.
    If i is unpaired, arr[i] = i (self-reference for safe gathering).
    """
    length = len(structure)
    partner_map = np.arange(length)  # Default to self
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


def one_hot_encode(seq, mapping, depth):
    """
    One-hot encodes a sequence string based on a mapping.
    """
    indices = [mapping.get(char, 0) for char in seq]
    # Create one-hot
    one_hot = np.eye(depth)[indices]
    return one_hot


def process_dataframe(df, is_test=False):
    """
    Processes a dataframe to generate features and targets.
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LENGTH

    # Feature dimensions
    # Seq (4) + Struct (3) + Loop (7) + PartnerID (4) = 18
    input_dim = 4 + 3 + 7 + 4

    # Initialize arrays
    inputs = np.zeros((num_samples, seq_len, input_dim), dtype=np.float32)
    partner_indices = np.zeros((num_samples, seq_len), dtype=np.int64)
    ids = df["id"].values

    # Initialize targets (only for train/val)
    # 5 target columns
    targets = np.zeros((num_samples, seq_len, 5), dtype=np.float32)

    target_cols = Config.TARGET_COLS

    for idx, row in df.iterrows():
        # 1. Parse Sequences
        sequence = row["sequence"]
        structure = row["structure"]
        loop_type = row["predicted_loop_type"]

        # 2. Base One-Hot Encodings
        oh_seq = one_hot_encode(sequence, SEQ_MAP, 4)
        oh_struct = one_hot_encode(structure, STRUCT_MAP, 3)
        oh_loop = one_hot_encode(loop_type, LOOP_MAP, 7)

        # 3. Partner Logic
        p_map = get_partner_map(structure)
        partner_indices[idx] = p_map

        # 4. Partner Identity Feature
        # If paired (p_map[i] != i), get one-hot of partner. Else zeros.
        oh_partner = np.zeros((seq_len, 4), dtype=np.float32)
        for i in range(seq_len):
            j = p_map[i]
            if i != j:  # Paired
                # The partner base is sequence[j]
                # We can reuse the oh_seq we just computed
                oh_partner[i] = oh_seq[j]
            # Else unpaired, remains 0

        # 5. Concatenate Inputs
        # Shape: (107, 18)
        sample_input = np.concatenate([oh_seq, oh_struct, oh_loop, oh_partner], axis=1)
        inputs[idx] = sample_input

        # 6. Process Targets (if not test)
        if not is_test:
            for t_i, col in enumerate(target_cols):
                val_str = row[col]
                try:
                    val_list = ast.literal_eval(val_str)
                except:
                    val_list = []

                # Fill the scored part
                length_scored = len(val_list)
                if length_scored > 0:
                    targets[idx, :length_scored, t_i] = val_list

    return inputs, partner_indices, targets, ids


def process_and_cache_data(csv_path, cache_path, is_test=False, load_cached_data=True):
    """
    Checks for cached .npz file. If not found or forced reload,
    processes the CSV and saves the cache.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        data = np.load(cache_path, allow_pickle=True)
        return data["inputs"], data["partner_indices"], data["targets"], data["ids"]

    print(f"Processing data from {csv_path}...")
    df = pd.read_csv(csv_path)

    inputs, partner_indices, targets, ids = process_dataframe(df, is_test=is_test)

    print(f"Saving data to {cache_path}...")
    np.savez_compressed(
        cache_path,
        inputs=inputs,
        partner_indices=partner_indices,
        targets=targets,
        ids=ids,
    )

    return inputs, partner_indices, targets, ids


class RNADataset(Dataset):
    def __init__(self, inputs, partner_indices, targets, ids, is_test=False):
        self.inputs = inputs
        self.partner_indices = partner_indices
        self.targets = targets
        self.ids = ids
        self.is_test = is_test

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Convert to torch tensors
        inp = torch.tensor(self.inputs[idx], dtype=torch.float32)
        p_idx = torch.tensor(self.partner_indices[idx], dtype=torch.long)

        if self.is_test:
            # Dummy targets for test
            tgt = torch.zeros((Config.SEQ_LENGTH, 5), dtype=torch.float32)
        else:
            tgt = torch.tensor(self.targets[idx], dtype=torch.float32)

        return inp, p_idx, tgt


def get_loaders(load_cached_data=True):
    """
    Main entry point to get DataLoaders.
    Handles caching and dataset creation.
    """
    Config.setup()

    # 1. Train Data
    train_inputs, train_p_idx, train_targets, train_ids = process_and_cache_data(
        Config.TRAIN_CSV,
        Config.TRAIN_DATA_PATH,
        is_test=False,
        load_cached_data=load_cached_data,
    )

    # 2. Val Data
    val_inputs, val_p_idx, val_targets, val_ids = process_and_cache_data(
        Config.VAL_CSV,
        Config.VAL_DATA_PATH,
        is_test=False,
        load_cached_data=load_cached_data,
    )

    # 3. Test Data
    test_inputs, test_p_idx, test_targets, test_ids = process_and_cache_data(
        Config.TEST_CSV,
        Config.TEST_DATA_PATH,
        is_test=True,
        load_cached_data=load_cached_data,
    )

    # 4. Create Datasets
    train_dataset = RNADataset(
        train_inputs, train_p_idx, train_targets, train_ids, is_test=False
    )
    val_dataset = RNADataset(val_inputs, val_p_idx, val_targets, val_ids, is_test=False)
    test_dataset = RNADataset(
        test_inputs, test_p_idx, test_targets, test_ids, is_test=True
    )

    # 5. Create Loaders
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
