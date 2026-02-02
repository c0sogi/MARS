import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA degradation prediction.
    Yields:
        x: (Channels, Seq_Len) - Input features including sequence, structure, loop, and partner identity.
        partner_indices: (Seq_Len,) - Indices of paired bases for latent gathering.
        y: (Seq_Len, Num_Targets) - Ground truth targets (or zeros for test set).
    """

    def __init__(self, inputs, partner_indices, targets=None, ids=None):
        self.inputs = inputs
        self.partner_indices = partner_indices
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # inputs shape: (Channels, Seq_Len)
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)

        # partner_indices shape: (Seq_Len,)
        p_idx = torch.tensor(self.partner_indices[idx], dtype=torch.long)

        if self.targets is not None:
            # targets shape: (Seq_Len, Num_Targets)
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
        else:
            # Dummy targets for inference
            y = torch.zeros((Config.SEQ_LEN, Config.NUM_TARGETS), dtype=torch.float32)

        return x, p_idx, y


def get_structure_indices(structure):
    """
    Parses dot-bracket structure string to find pairing partners.
    Args:
        structure (str): Dot-bracket notation string (e.g., "((..))").
    Returns:
        np.ndarray: Array of shape (Seq_Len,) where arr[i] is the index of the partner of base i.
                    Unpaired bases are marked with -1.
    """
    length = len(structure)
    partners = np.full(length, -1, dtype=int)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                partners[i] = j
                partners[j] = i
    return partners


def one_hot_encode(seq, vocab):
    """
    One-hot encodes a sequence based on a vocabulary list.
    Args:
        seq (str): Input sequence.
        vocab (list): List of allowed characters.
    Returns:
        np.ndarray: One-hot encoded matrix of shape (Seq_Len, Vocab_Size).
    """
    mapping = {char: i for i, char in enumerate(vocab)}
    seq_len = len(seq)
    vocab_size = len(vocab)
    one_hot = np.zeros((seq_len, vocab_size), dtype=np.float32)

    for i, char in enumerate(seq):
        if char in mapping:
            one_hot[i, mapping[char]] = 1.0
    return one_hot


def process_data(csv_path, mode="train"):
    """
    Reads CSV metadata, generates features, and prepares targets.
    Args:
        csv_path (str): Path to the CSV file.
        mode (str): 'train', 'val', or 'test'.
    Returns:
        tuple: (inputs, partner_indices, targets, ids)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Vocabularies
    seq_vocab = ["A", "G", "C", "U"]
    struct_vocab = ["(", ")", "."]
    loop_vocab = ["S", "M", "I", "B", "H", "E", "X"]

    all_inputs = []
    all_partner_indices = []
    all_targets = []
    all_ids = df["id"].values

    # Check for targets
    target_cols = Config.TARGET_COLS
    has_targets = all(col in df.columns for col in target_cols)

    # Parse stringified lists in target columns if they exist
    if has_targets:
        for col in target_cols:
            # Use ast.literal_eval to safely parse string representation of lists
            df[col] = df[col].apply(
                lambda x: (
                    np.array(ast.literal_eval(x), dtype=np.float32)
                    if isinstance(x, str)
                    else x
                )
            )

    for idx, row in df.iterrows():
        sequence = row["sequence"]
        structure = row["structure"]
        loop_type = row["predicted_loop_type"]

        # --- Feature Generation ---

        # 1. Standard One-Hot Features
        # Shapes: (Seq_Len, Vocab_Size)
        feat_seq = one_hot_encode(sequence, seq_vocab)  # 4 channels
        feat_struct = one_hot_encode(structure, struct_vocab)  # 3 channels
        feat_loop = one_hot_encode(loop_type, loop_vocab)  # 7 channels

        # 2. Partner Indices
        p_indices = get_structure_indices(structure)

        # 3. Partner Identity (Input-Level Context)
        # Create a (Seq_Len, 4) matrix where row i contains the one-hot vector of the base paired with i
        feat_partner = np.zeros((len(sequence), 4), dtype=np.float32)
        for i, p_idx in enumerate(p_indices):
            if p_idx != -1:
                feat_partner[i] = feat_seq[p_idx]

        # Concatenate all features along the channel dimension
        # Result: (Seq_Len, 4 + 3 + 7 + 4) = (Seq_Len, 18)
        combined_features = np.concatenate(
            [feat_seq, feat_struct, feat_loop, feat_partner], axis=1
        )

        # Transpose to (Channels, Seq_Len) for PyTorch Conv1d compatibility
        combined_features = combined_features.transpose(1, 0)

        all_inputs.append(combined_features)
        all_partner_indices.append(p_indices)

        # --- Target Generation ---
        if has_targets:
            # Initialize target matrix (Seq_Len, Num_Targets)
            target_matrix = np.zeros(
                (Config.SEQ_LEN, Config.NUM_TARGETS), dtype=np.float32
            )

            for t_i, col in enumerate(target_cols):
                val_arr = row[col]
                # Fill valid length (e.g., first 68 bases)
                # Pad the rest with zeros (masked out by loss function later)
                length = min(len(val_arr), Config.SEQ_LEN)
                target_matrix[:length, t_i] = val_arr[:length]

            all_targets.append(target_matrix)

    # Convert lists to numpy arrays
    all_inputs = np.array(all_inputs, dtype=np.float32)
    all_partner_indices = np.array(all_partner_indices, dtype=np.int64)

    if has_targets:
        all_targets = np.array(all_targets, dtype=np.float32)
    else:
        all_targets = None

    return all_inputs, all_partner_indices, all_targets, all_ids


def get_loaders(load_cached_data=True):
    """
    Generates DataLoaders for train, validation, and test sets.
    Handles caching of processed numpy arrays to speed up subsequent runs.

    Args:
        load_cached_data (bool): If True, attempts to load from .npz cache files.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    train_cache = Config.TRAIN_CACHE
    val_cache = Config.VAL_CACHE
    test_cache = Config.TEST_CACHE

    # --- Load/Process Train Data ---
    if load_cached_data and os.path.exists(train_cache):
        # print(f"Loading train data from cache: {train_cache}")
        data = np.load(train_cache, allow_pickle=True)
        train_inputs = data["inputs"]
        train_p_idx = data["partner_indices"]
        train_targets = data["targets"]
        train_ids = data["ids"]
    else:
        # print("Processing train data from CSV...")
        train_inputs, train_p_idx, train_targets, train_ids = process_data(
            Config.TRAIN_CSV, mode="train"
        )
        np.savez_compressed(
            train_cache,
            inputs=train_inputs,
            partner_indices=train_p_idx,
            targets=train_targets,
            ids=train_ids,
        )

    # --- Load/Process Val Data ---
    if load_cached_data and os.path.exists(val_cache):
        # print(f"Loading val data from cache: {val_cache}")
        data = np.load(val_cache, allow_pickle=True)
        val_inputs = data["inputs"]
        val_p_idx = data["partner_indices"]
        val_targets = data["targets"]
        val_ids = data["ids"]
    else:
        # print("Processing val data from CSV...")
        val_inputs, val_p_idx, val_targets, val_ids = process_data(
            Config.VAL_CSV, mode="val"
        )
        np.savez_compressed(
            val_cache,
            inputs=val_inputs,
            partner_indices=val_p_idx,
            targets=val_targets,
            ids=val_ids,
        )

    # --- Load/Process Test Data ---
    if load_cached_data and os.path.exists(test_cache):
        # print(f"Loading test data from cache: {test_cache}")
        data = np.load(test_cache, allow_pickle=True)
        test_inputs = data["inputs"]
        test_p_idx = data["partner_indices"]
        test_ids = data["ids"]
        test_targets = None
    else:
        # print("Processing test data from CSV...")
        test_inputs, test_p_idx, test_targets, test_ids = process_data(
            Config.TEST_CSV, mode="test"
        )
        np.savez_compressed(
            test_cache, inputs=test_inputs, partner_indices=test_p_idx, ids=test_ids
        )

    # --- Create Datasets ---
    train_dataset = RNADataset(train_inputs, train_p_idx, train_targets, train_ids)
    val_dataset = RNADataset(val_inputs, val_p_idx, val_targets, val_ids)
    test_dataset = RNADataset(test_inputs, test_p_idx, targets=None, ids=test_ids)

    # --- Create Loaders ---
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
