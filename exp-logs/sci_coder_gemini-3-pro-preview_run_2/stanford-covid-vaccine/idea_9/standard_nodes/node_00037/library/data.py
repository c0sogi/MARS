import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed

# =============================================================================
# Helper Functions for Feature Generation
# =============================================================================


def get_structure_indices(structure):
    """
    Parses a dot-bracket structure string to find paired indices.
    Returns a numpy array of length L, where arr[i] is the index of the partner
    of base i. If base i is unpaired, arr[i] is -1.
    """
    n = len(structure)
    partner_indices = np.full(n, -1, dtype=np.int32)
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


def sequence_to_one_hot(seq):
    """
    Converts sequence string to one-hot encoding (L, 4).
    Map: A:0, G:1, C:2, U:3
    """
    mapping = {"A": 0, "G": 1, "C": 2, "U": 3}
    L = len(seq)
    one_hot = np.zeros((L, 4), dtype=np.float32)
    for i, char in enumerate(seq):
        if char in mapping:
            one_hot[i, mapping[char]] = 1.0
    return one_hot


def structure_to_one_hot(struct):
    """
    Converts structure string to one-hot encoding (L, 3).
    Map: .:0, (:1, ):2
    """
    mapping = {".": 0, "(": 1, ")": 2}
    L = len(struct)
    one_hot = np.zeros((L, 3), dtype=np.float32)
    for i, char in enumerate(struct):
        if char in mapping:
            one_hot[i, mapping[char]] = 1.0
    return one_hot


def loop_type_to_one_hot(loop_str):
    """
    Converts predicted loop type string to one-hot encoding (L, 7).
    Map: S:0, M:1, I:2, B:3, H:4, E:5, X:6
    """
    mapping = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}
    L = len(loop_str)
    one_hot = np.zeros((L, 7), dtype=np.float32)
    for i, char in enumerate(loop_str):
        if char in mapping:
            one_hot[i, mapping[char]] = 1.0
    return one_hot


def get_partner_identity_one_hot(sequence, partner_indices):
    """
    Creates the Partner Base Identity feature (L, 5).
    For each position i, if paired with j, encodes sequence[j].
    If unpaired, encodes 'None' (index 4).
    Map: A:0, G:1, C:2, U:3, None:4
    """
    seq_map = {"A": 0, "G": 1, "C": 2, "U": 3}
    L = len(sequence)
    one_hot = np.zeros((L, 5), dtype=np.float32)

    for i in range(L):
        partner_idx = partner_indices[i]
        if partner_idx != -1:
            # Paired: use identity of partner
            partner_base = sequence[partner_idx]
            if partner_base in seq_map:
                one_hot[i, seq_map[partner_base]] = 1.0
        else:
            # Unpaired: use 'None' class
            one_hot[i, 4] = 1.0

    return one_hot


def parse_target_column(col_data, seq_len=107):
    """
    Parses a pandas Series of stringified lists into a padded numpy array.
    """

    # Convert strings to lists
    # Handle cases where data is already list (rare but possible in some pipelines)
    def parse_item(x):
        if isinstance(x, str):
            try:
                return ast.literal_eval(x)
            except:
                return []
        return x if isinstance(x, list) else []

    parsed = col_data.apply(parse_item).tolist()

    # Pad to seq_len
    # Targets are usually length 68, need to pad to 107
    padded = np.zeros((len(parsed), seq_len), dtype=np.float32)
    for i, row in enumerate(parsed):
        length = len(row)
        if length > 0:
            # Copy available data
            # Truncate if longer than seq_len (unlikely for this dataset)
            valid_len = min(length, seq_len)
            padded[i, :valid_len] = row[:valid_len]

    return padded


# =============================================================================
# Data Processing Logic
# =============================================================================


def process_dataframe(df, is_test=False):
    """
    Converts a dataframe into numpy arrays for inputs, partner_indices, and targets.
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LENGTH

    # Initialize arrays
    # Input channels: 4 (Seq) + 3 (Struct) + 7 (Loop) + 5 (PartnerID) = 19
    inputs = np.zeros((num_samples, seq_len, Config.INPUT_CHANNELS), dtype=np.float32)
    partner_indices_arr = np.zeros((num_samples, seq_len), dtype=np.int32)

    # Targets: 5 columns
    if not is_test:
        targets = np.zeros((num_samples, seq_len, Config.NUM_TARGETS), dtype=np.float32)
    else:
        targets = None

    ids = df["id"].values

    # Process features row by row
    for idx, row in df.iterrows():
        # 0. Extract raw strings
        seq = row["sequence"]
        struct = row["structure"]
        loop = row["predicted_loop_type"]

        # 1. Basic One-Hots
        oh_seq = sequence_to_one_hot(seq)  # (L, 4)
        oh_struct = structure_to_one_hot(struct)  # (L, 3)
        oh_loop = loop_type_to_one_hot(loop)  # (L, 7)

        # 2. Partner Info
        p_indices = get_structure_indices(struct)
        oh_partner = get_partner_identity_one_hot(seq, p_indices)  # (L, 5)

        # 3. Concatenate Inputs
        # Order: Seq, Struct, Loop, PartnerID
        sample_input = np.concatenate([oh_seq, oh_struct, oh_loop, oh_partner], axis=1)

        # Store
        # Since df index might not be contiguous 0..N, use enumeration index if needed
        # But here we initialized arrays by size, so we need a linear index.
        # Let's assume df is reset_index or we use enumerate on df.itertuples/iterrows
        # Safer to use a separate counter
        pass

    # Re-loop with explicit index
    for i in range(num_samples):
        row = df.iloc[i]
        seq = row["sequence"]
        struct = row["structure"]
        loop = row["predicted_loop_type"]

        oh_seq = sequence_to_one_hot(seq)
        oh_struct = structure_to_one_hot(struct)
        oh_loop = loop_type_to_one_hot(loop)
        p_indices = get_structure_indices(struct)
        oh_partner = get_partner_identity_one_hot(seq, p_indices)

        inputs[i] = np.concatenate([oh_seq, oh_struct, oh_loop, oh_partner], axis=1)
        partner_indices_arr[i] = p_indices

    # Process Targets
    if not is_test:
        for t_idx, col in enumerate(Config.TARGET_COLS):
            # parse_target_column returns (N, L)
            col_data = parse_target_column(df[col], seq_len)
            targets[:, :, t_idx] = col_data

    return inputs, partner_indices_arr, targets, ids


def load_or_process_data(csv_path, cache_path, load_cached_data=True, is_test=False):
    """
    Loads data from cache if available, otherwise processes from CSV and caches it.
    """
    # Ensure working directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            data = np.load(cache_path, allow_pickle=True)
            inputs = data["inputs"]
            partner_indices = data["partner_indices"]
            ids = data["ids"]
            if is_test:
                targets = None
            else:
                targets = data["targets"]
            return inputs, partner_indices, targets, ids
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    print(f"Processing data from {csv_path}...")
    df = pd.read_csv(csv_path)
    inputs, partner_indices, targets, ids = process_dataframe(df, is_test=is_test)

    print(f"Saving processed data to {cache_path}...")
    if is_test:
        np.savez_compressed(
            cache_path, inputs=inputs, partner_indices=partner_indices, ids=ids
        )
    else:
        np.savez_compressed(
            cache_path,
            inputs=inputs,
            partner_indices=partner_indices,
            targets=targets,
            ids=ids,
        )

    return inputs, partner_indices, targets, ids


# =============================================================================
# Dataset Class
# =============================================================================


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA Degradation Prediction.
    Yields:
        inputs: (L, C) tensor
        partner_indices: (L,) tensor (unpaired mapped to self)
        targets: (L, T) tensor
    """

    def __init__(self, inputs, partner_indices, targets=None, ids=None):
        self.inputs = inputs
        self.partner_indices = partner_indices
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Convert to tensors
        input_tensor = torch.from_numpy(self.inputs[idx])

        # Handle Partner Indices
        # Map -1 (unpaired) to the current index (self) for safe gathering
        p_indices = self.partner_indices[idx].copy()
        unpaired_mask = p_indices == -1
        # Create an array of indices [0, 1, ..., L-1]
        self_indices = np.arange(len(p_indices), dtype=np.int32)
        # Replace -1 with self index
        p_indices[unpaired_mask] = self_indices[unpaired_mask]

        partner_idx_tensor = torch.from_numpy(p_indices).long()

        if self.targets is not None:
            target_tensor = torch.from_numpy(self.targets[idx])
        else:
            # Dummy targets for test set
            target_tensor = torch.zeros(
                (input_tensor.shape[0], Config.NUM_TARGETS), dtype=torch.float32
            )

        return input_tensor, partner_idx_tensor, target_tensor


# =============================================================================
# Data Loader Factory
# =============================================================================


def get_dataloaders(load_cached_data=True, batch_size=None, num_workers=None):
    """
    Creates DataLoaders for Train, Val, and Test sets.
    """
    # Use Config defaults if not provided
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    if num_workers is None:
        num_workers = Config.NUM_WORKERS

    # 1. Load Data
    train_inputs, train_p_idx, train_targets, _ = load_or_process_data(
        Config.TRAIN_CSV, Config.TRAIN_CACHE, load_cached_data, is_test=False
    )

    val_inputs, val_p_idx, val_targets, _ = load_or_process_data(
        Config.VAL_CSV, Config.VAL_CACHE, load_cached_data, is_test=False
    )

    test_inputs, test_p_idx, _, test_ids = load_or_process_data(
        Config.TEST_CSV, Config.TEST_CACHE, load_cached_data, is_test=True
    )

    # 2. Create Datasets
    train_dataset = RNADataset(train_inputs, train_p_idx, train_targets)
    val_dataset = RNADataset(val_inputs, val_p_idx, val_targets)
    test_dataset = RNADataset(test_inputs, test_p_idx, None, ids=test_ids)

    # 3. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for training stability
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

    return train_loader, val_loader, test_loader, test_ids
