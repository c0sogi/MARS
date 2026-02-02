import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import PATHS, DATA_CONFIG, TRAIN_PARAMS


def get_structure_indices(structure):
    """
    Parses dot-bracket structure to find partner indices.
    Returns an array of length L where arr[i] is the index of the partner of i.
    Unpaired positions are marked with -1.
    """
    n = len(structure)
    indices = np.full(n, -1, dtype=int)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                indices[i] = j
                indices[j] = i
    return indices


def get_one_hot(sequence, vocab):
    """
    Converts a sequence string into a one-hot encoded numpy array.
    Returns shape (L, len(vocab)).
    """
    char_to_idx = {char: i for i, char in enumerate(vocab)}
    indices = [char_to_idx.get(c, -1) for c in sequence]

    one_hot = np.zeros((len(sequence), len(vocab)), dtype=np.float32)
    for i, idx in enumerate(indices):
        if idx != -1:
            one_hot[i, idx] = 1.0
    return one_hot


def process_data(csv_path, cache_path, load_cached_data=True, is_test=False):
    """
    Loads data from CSV, generates features, and handles caching.

    Args:
        csv_path (str): Path to the source CSV file.
        cache_path (str): Path to the .npz cache file.
        load_cached_data (bool): Whether to attempt loading from cache.
        is_test (bool): Whether processing test data (no targets).

    Returns:
        Tuple of numpy arrays: (inputs, partner_indices, targets, masks, ids)
    """
    # 1. Attempt to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        try:
            data = np.load(cache_path, allow_pickle=True)
            return (
                data["inputs"],
                data["partner_indices"],
                data["targets"],
                data["masks"],
                data["ids"],
            )
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing from source.")

    # 2. Process from source
    print(f"Processing data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Configuration
    vocab_bases = DATA_CONFIG["vocab_bases"]
    vocab_struct = DATA_CONFIG["vocab_structure"]
    vocab_loop = DATA_CONFIG["vocab_loop"]
    seq_len = DATA_CONFIG["seq_length"]
    scored_len = DATA_CONFIG["scored_length"]
    target_cols = DATA_CONFIG["target_cols"]

    # Containers
    all_inputs = []
    all_partner_indices = []
    all_targets = []
    all_masks = []
    all_ids = []

    for _, row in df.iterrows():
        # --- Feature Generation ---
        seq = row["sequence"]
        struct = row["structure"]
        loop = row["predicted_loop_type"]

        # Base One-Hot Encodings
        oh_seq = get_one_hot(seq, vocab_bases)  # (L, 4)
        oh_struct = get_one_hot(struct, vocab_struct)  # (L, 3)
        oh_loop = get_one_hot(loop, vocab_loop)  # (L, 7)

        # Partner Context
        p_indices = get_structure_indices(struct)

        # Partner Identity: If i is paired with j, use oh_seq[j]. Else zeros.
        oh_partner = np.zeros_like(oh_seq)
        for i, p_idx in enumerate(p_indices):
            if p_idx != -1:
                oh_partner[i] = oh_seq[p_idx]

        # Concatenate Static Features (18 channels)
        # Seq(4) + Struct(3) + Loop(7) + PartnerID(4)
        sample_inputs = np.concatenate([oh_seq, oh_struct, oh_loop, oh_partner], axis=1)

        # Sanitize partner indices for tensor usage (replace -1 with 0)
        # The model should mask unpaired positions using the structure input.
        p_indices_sanitized = p_indices.copy()
        p_indices_sanitized[p_indices == -1] = 0

        # --- Target & Mask Generation ---
        sample_targets = np.zeros((seq_len, len(target_cols)), dtype=np.float32)
        sample_mask = np.zeros(seq_len, dtype=np.float32)

        # Mask is active for the scored length
        sample_mask[:scored_len] = 1.0

        if not is_test:
            for t_i, col in enumerate(target_cols):
                val_str = row[col]
                try:
                    # Targets are stored as stringified lists
                    vals = ast.literal_eval(val_str)
                except (ValueError, SyntaxError):
                    vals = []

                # Copy values up to the available length (usually scored_len)
                length = min(len(vals), seq_len)
                sample_targets[:length, t_i] = vals[:length]

        all_inputs.append(sample_inputs)
        all_partner_indices.append(p_indices_sanitized)
        all_targets.append(sample_targets)
        all_masks.append(sample_mask)
        all_ids.append(row["id"])

    # Convert to numpy arrays
    all_inputs = np.array(all_inputs, dtype=np.float32)
    all_partner_indices = np.array(all_partner_indices, dtype=np.int32)
    all_targets = np.array(all_targets, dtype=np.float32)
    all_masks = np.array(all_masks, dtype=np.float32)
    all_ids = np.array(all_ids)

    # 3. Save to Cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        inputs=all_inputs,
        partner_indices=all_partner_indices,
        targets=all_targets,
        masks=all_masks,
        ids=all_ids,
    )
    print(f"Saved processed data to {cache_path}")

    return all_inputs, all_partner_indices, all_targets, all_masks, all_ids


class RNADataset(Dataset):
    def __init__(self, inputs, partner_indices, targets, masks, ids):
        self.inputs = inputs
        self.partner_indices = partner_indices
        self.targets = targets
        self.masks = masks
        self.ids = ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.inputs[idx], dtype=torch.float32),
            torch.tensor(self.partner_indices[idx], dtype=torch.long),
            torch.tensor(self.targets[idx], dtype=torch.float32),
            torch.tensor(self.masks[idx], dtype=torch.float32),
            self.ids[idx],
        )


def get_dataloaders(debug=False, load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        debug (bool): If True, limits the dataset size for debugging.
        load_cached_data (bool): Whether to use cached .npz files.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load/Process Data
    # We process the full dataset first to ensure the cache is complete and valid.
    train_inputs, train_pi, train_targets, train_masks, train_ids = process_data(
        PATHS["TRAIN_CSV"], PATHS["TRAIN_CACHE"], load_cached_data, is_test=False
    )

    val_inputs, val_pi, val_targets, val_masks, val_ids = process_data(
        PATHS["VAL_CSV"], PATHS["VAL_CACHE"], load_cached_data, is_test=False
    )

    test_inputs, test_pi, test_targets, test_masks, test_ids = process_data(
        PATHS["TEST_CSV"], PATHS["TEST_CACHE"], load_cached_data, is_test=True
    )

    # Debug Subsetting
    if debug:
        limit = TRAIN_PARAMS.get("max_debug_samples", 100)
        print(f"Debug mode: limiting data to {limit} samples.")

        train_inputs = train_inputs[:limit]
        train_pi = train_pi[:limit]
        train_targets = train_targets[:limit]
        train_masks = train_masks[:limit]
        train_ids = train_ids[:limit]

        val_inputs = val_inputs[:limit]
        val_pi = val_pi[:limit]
        val_targets = val_targets[:limit]
        val_masks = val_masks[:limit]
        val_ids = val_ids[:limit]

        test_inputs = test_inputs[:limit]
        test_pi = test_pi[:limit]
        test_targets = test_targets[:limit]
        test_masks = test_masks[:limit]
        test_ids = test_ids[:limit]

    # Create Datasets
    train_dataset = RNADataset(
        train_inputs, train_pi, train_targets, train_masks, train_ids
    )
    val_dataset = RNADataset(val_inputs, val_pi, val_targets, val_masks, val_ids)
    test_dataset = RNADataset(test_inputs, test_pi, test_targets, test_masks, test_ids)

    # Create Loaders
    batch_size = TRAIN_PARAMS["batch_size"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
