import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from library.config import Config


class RNADataset(Dataset):
    def __init__(self, sequences, loop_types, pair_dists, targets, masks, ids):
        """
        Args:
            sequences: (N, SEQ_LEN) LongTensor
            loop_types: (N, SEQ_LEN) LongTensor
            pair_dists: (N, SEQ_LEN, POS_EMBED_DIM) FloatTensor
            targets: (N, SEQ_LEN, 3) FloatTensor
            masks: (N, SEQ_LEN) BoolTensor
            ids: List of sequence IDs
        """
        self.sequences = sequences
        self.loop_types = loop_types
        self.pair_dists = pair_dists
        self.targets = targets
        self.masks = masks
        self.ids = ids

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            "sequence": self.sequences[idx],
            "loop_type": self.loop_types[idx],
            "pair_dist": self.pair_dists[idx],
            "targets": self.targets[idx],
            "mask": self.masks[idx],
            "id": self.ids[idx],
        }


def process_structure(structure_str, seq_len):
    """
    Parses dot-bracket structure to compute signed pairing distances.
    Returns: numpy array of shape (seq_len,) containing signed distances.
    """
    stack = []
    dists = np.zeros(seq_len, dtype=np.float32)

    for i, char in enumerate(structure_str):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                # Pair (j, i) where j < i
                # Distance for j is i - j (positive)
                # Distance for i is j - i (negative)
                dists[j] = i - j
                dists[i] = j - i
    return dists


def compute_sinusoidal_encoding(dists, dim):
    """
    Computes sinusoidal positional encoding for signed distances.
    Args:
        dists: (N, L) numpy array of distances
        dim: embedding dimension
    Returns:
        (N, L, dim) numpy array
    """
    N, L = dists.shape
    pe = np.zeros((N, L, dim), dtype=np.float32)

    # Geometric progression of frequencies
    div_term = np.exp(np.arange(0, dim, 2) * -(np.log(10000.0) / dim))

    # dists: (N, L) -> (N, L, 1)
    pos_expanded = dists[:, :, None]
    # div_term: (dim/2,) -> (1, 1, dim/2)
    div_expanded = div_term[None, None, :]

    # Calculate arguments
    args = pos_expanded * div_expanded

    # Apply sin to even indices, cos to odd indices
    pe[:, :, 0::2] = np.sin(args)
    pe[:, :, 1::2] = np.cos(args)

    return pe


def load_data(split="train", debug=False, load_cached_data=True):
    """
    Loads data, processes features, and returns an RNADataset.
    Handles caching to ./working/idea_21/.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    cache_path = os.path.join(Config.WORKING_DIR, f"full_{split}_data.pt")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {split} data from {cache_path}...")
        data_dict = torch.load(cache_path)

        # Handle Debug Slicing on Cached Data
        if debug:
            limit = Config.DEBUG_SUBSET_SIZE
            print(f"Debug mode: Slicing first {limit} samples.")
            return RNADataset(
                sequences=data_dict["sequences"][:limit],
                loop_types=data_dict["loop_types"][:limit],
                pair_dists=data_dict["pair_dists"][:limit],
                targets=data_dict["targets"][:limit],
                masks=data_dict["masks"][:limit],
                ids=data_dict["ids"][:limit],
            )
        else:
            return RNADataset(**data_dict)

    # 2. Process from scratch
    print(f"Processing {split} data from Parquet...")

    # Identify source file
    if split == "train":
        path = Config.TRAIN_PATH
    elif split == "val":
        path = Config.VAL_PATH
    elif split == "test":
        path = Config.TEST_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    df = pd.read_parquet(path)

    # Handle Debug Slicing on Raw Data (if not caching, we slice early for speed)
    # However, to maintain cache consistency, we only slice early if we are NOT going to save the cache.
    # But the requirement says "implement caching". So we process full, save, then slice if debug.
    # To save time in active development, if debug is True and cache missing, we process only subset and DO NOT save.
    if debug:
        print(f"Debug mode: Processing only {Config.DEBUG_SUBSET_SIZE} samples.")
        df = df.iloc[: Config.DEBUG_SUBSET_SIZE].copy()

    # --- Feature Extraction ---

    # 1. Sequences
    # Map chars to IDs: A->0, G->1, C->2, U->3
    seq_list = []
    for seq in df["sequence"]:
        seq_ids = [
            Config.TOKEN_TO_ID.get(c, 0) for c in seq
        ]  # Default to 0 if unknown, though vocab is fixed
        seq_list.append(seq_ids)
    sequences = torch.tensor(seq_list, dtype=torch.long)

    # 2. Loop Types
    # Map chars to IDs
    loop_list = []
    for loop in df["predicted_loop_type"]:
        loop_ids = [Config.LOOP_TO_ID.get(c, 0) for c in loop]
        loop_list.append(loop_ids)
    loop_types = torch.tensor(loop_list, dtype=torch.long)

    # 3. Geometric Encoding (Signed Sinusoidal Pairing Distance)
    # Parse structure strings
    dist_list = []
    for struct in df["structure"]:
        dists = process_structure(struct, Config.SEQ_LEN)
        dist_list.append(dists)

    dist_array = np.array(dist_list)  # (N, 107)
    # Compute embeddings
    pair_dists_np = compute_sinusoidal_encoding(dist_array, Config.POS_EMBED_DIM)
    pair_dists = torch.tensor(pair_dists_np, dtype=torch.float32)

    # 4. Targets & Masks
    # Initialize containers
    N = len(df)
    L = Config.SEQ_LEN
    n_targets = len(Config.TARGET_COLS)

    targets_tensor = torch.zeros((N, L, n_targets), dtype=torch.float32)
    masks_tensor = torch.zeros((N, L), dtype=torch.bool)

    if split != "test":
        # Extract ground truth
        # Columns are lists of length 68
        for i, col in enumerate(Config.TARGET_COLS):
            # Convert column of lists to numpy array
            # Note: df[col] contains lists. np.vstack stacks them.
            col_data = np.vstack(df[col].values)  # Shape (N, 68)

            # Place into tensor
            # We assume the first 68 positions are the scored ones
            targets_tensor[:, : Config.PRED_LEN, i] = torch.tensor(
                col_data, dtype=torch.float32
            )

        # Create mask (True for first 68 positions)
        masks_tensor[:, : Config.PRED_LEN] = True
    else:
        # Test set has no targets, keep zeros.
        # Mask is all false (or irrelevant), keep False.
        pass

    ids = df["id"].tolist()

    # Construct Data Dictionary
    data_dict = {
        "sequences": sequences,
        "loop_types": loop_types,
        "pair_dists": pair_dists,
        "targets": targets_tensor,
        "masks": masks_tensor,
        "ids": ids,
    }

    # Save to cache ONLY if processed full dataset (not debug)
    if not debug:
        print(f"Saving processed data to {cache_path}...")
        torch.save(data_dict, cache_path)

    return RNADataset(**data_dict)
