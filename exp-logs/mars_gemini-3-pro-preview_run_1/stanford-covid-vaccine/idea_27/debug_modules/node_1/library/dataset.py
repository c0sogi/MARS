import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA Degradation Prediction.
    Implements Synchronized Structural Augmentation.
    """

    def __init__(
        self, sequences, loop_types, pair_indices, targets=None, ids=None, augment=False
    ):
        """
        Args:
            sequences (np.ndarray): (N, Seq_Len) Int array of sequence tokens.
            loop_types (np.ndarray): (N, Seq_Len) Int array of loop type tokens.
            pair_indices (np.ndarray): (N, Seq_Len) Int array where value at i is the index of its pair partner, or -1.
            targets (np.ndarray, optional): (N, Pred_Len, Num_Targets) Float array of ground truth.
            ids (list/np.ndarray, optional): List of sample IDs.
            augment (bool): Whether to apply structural augmentation.
        """
        self.sequences = sequences
        self.loop_types = loop_types
        self.pair_indices = pair_indices
        self.targets = targets
        self.ids = ids
        self.augment = augment

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Clone data to avoid modifying shared arrays
        seq = self.sequences[idx]  # (107,)
        loop = self.loop_types[idx].copy()  # (107,)
        partners = self.pair_indices[idx].copy()  # (107,)

        # Synchronized Structural Augmentation
        if self.augment:
            # Identify unique pairs to drop
            # We look for indices i where partners[i] > i to get each pair (i, j) exactly once
            opening_mask = (partners > -1) & (np.arange(len(partners)) < partners)
            opening_indices = np.where(opening_mask)[0]

            if len(opening_indices) > 0:
                # Vectorized coin flip for all pairs in this sequence
                probs = np.random.rand(len(opening_indices))
                drop_mask = probs < Config.AUG_PROB
                indices_to_drop = opening_indices[drop_mask]

                for i in indices_to_drop:
                    j = partners[i]

                    # Synchronization:
                    # 1. Set Loop Type to 'X' (External/Unpaired)
                    loop[i] = Config.AUG_LOOP_ID
                    loop[j] = Config.AUG_LOOP_ID

                    # 2. Break the link (distance becomes 0 later)
                    partners[i] = -1
                    partners[j] = -1

        # Calculate Signed Pairing Distance
        # If paired: dist = partner_index - current_index
        # If unpaired (partners == -1): dist = 0
        distances = np.zeros_like(partners, dtype=np.int32)
        paired_mask = partners != -1
        # Example: i=0, j=106. partners[0]=106. dist = 106 - 0 = 106.
        # Example: i=106, j=0. partners[106]=0. dist = 0 - 106 = -106.
        distances[paired_mask] = (
            partners[paired_mask] - np.arange(len(partners))[paired_mask]
        )

        # Convert to Tensors
        # Sequence and Loop are LongTensors for Embedding layers
        seq_tensor = torch.tensor(seq, dtype=torch.long)
        loop_tensor = torch.tensor(loop, dtype=torch.long)

        # Distance is LongTensor (acting as indices for Sinusoidal Encoding or Embedding)
        dist_tensor = torch.tensor(distances, dtype=torch.long)

        item = {
            "sequence": seq_tensor,
            "loop_type": loop_tensor,
            "distance": dist_tensor,
        }

        if self.ids is not None:
            item["id"] = self.ids[idx]

        if self.targets is not None:
            # Targets are float32
            item["targets"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        return item


def parse_structure_string(structure_str, length):
    """
    Parses a dot-bracket structure string into a partner index array.
    Returns:
        np.ndarray: Array of length `length`. value at i is partner index j, or -1.
    """
    partners = np.full(length, -1, dtype=np.int16)
    stack = []
    for i, char in enumerate(structure_str):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                partners[i] = j
                partners[j] = i
    return partners


def process_data(mode, subset_size=None):
    """
    Reads Parquet file and processes features into numpy arrays.
    """
    # Select file based on mode
    if mode == "train":
        filepath = Config.TRAIN_FILE
    elif mode == "val":
        filepath = Config.VAL_FILE
    elif mode == "test":
        filepath = Config.TEST_FILE
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Load Parquet
    df = pd.read_parquet(filepath)

    # Subset if requested
    if subset_size is not None:
        df = df.iloc[:subset_size].copy()

    n_samples = len(df)
    seq_len = Config.SEQ_LEN

    # 1. Parse Sequences
    # Map chars to IDs
    sequences = np.zeros((n_samples, seq_len), dtype=np.int8)
    for i, seq in enumerate(df["sequence"]):
        sequences[i] = [Config.NUC_TO_ID.get(c, 0) for c in seq]

    # 2. Parse Loop Types
    loop_types = np.zeros((n_samples, seq_len), dtype=np.int8)
    for i, loop in enumerate(df["predicted_loop_type"]):
        loop_types[i] = [Config.LOOP_TO_ID.get(c, Config.LOOP_TO_ID["X"]) for c in loop]

    # 3. Parse Structures (Pair Indices)
    pair_indices = np.zeros((n_samples, seq_len), dtype=np.int16)
    for i, struct in enumerate(df["structure"]):
        pair_indices[i] = parse_structure_string(struct, seq_len)

    # 4. Parse Targets (if available)
    targets = None
    # Check if target columns exist (Test set won't have them)
    if all(col in df.columns for col in Config.TARGET_COLS):
        # Stack the specific target columns: reactivity, deg_Mg_pH10, deg_Mg_50C
        # Each cell in dataframe is a list/array of length 68
        target_arrays = []
        for col in Config.TARGET_COLS:
            # vstack converts column of lists to (N, 68) array
            col_data = np.vstack(df[col].values)
            target_arrays.append(col_data)

        # Stack along last dimension -> (N, 68, 3)
        targets = np.stack(target_arrays, axis=-1).astype(np.float32)

    # IDs
    ids = df["id"].values

    return {
        "sequences": sequences,
        "loop_types": loop_types,
        "pair_indices": pair_indices,
        "targets": targets,
        "ids": ids,
    }


def load_data(mode="train", load_cached_data=True, subset_size=None):
    """
    Main entry point to load data. Handles caching and dataset instantiation.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from .npz cache.
        subset_size (int, optional): If provided, limits data size. Disables caching.

    Returns:
        RNADataset: Instantiated dataset.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    cache_path = os.path.join(Config.WORKING_DIR, f"cached_{mode}.npz")

    # Disable caching if subsetting to avoid overwriting full cache with partial data
    if subset_size is not None:
        load_cached_data = False

    data = None

    # Try loading cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            # Allow pickle=True because we might store object arrays (strings for IDs),
            # though we prefer strict numpy types.
            loaded = np.load(cache_path, allow_pickle=True)

            # Reconstruct dictionary
            data = {
                "sequences": loaded["sequences"],
                "loop_types": loaded["loop_types"],
                "pair_indices": loaded["pair_indices"],
                "ids": loaded["ids"],
            }
            if "targets" in loaded:
                data["targets"] = loaded["targets"]
                # Handle None stored in npz (rare, usually key just missing)
                if data["targets"].ndim == 0:
                    data["targets"] = None
            else:
                data["targets"] = None

        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}. Re-processing.")
            data = None

    # Process if not loaded
    if data is None:
        data = process_data(mode, subset_size=subset_size)

        # Save to cache (only if full set)
        if subset_size is None:
            save_dict = {
                "sequences": data["sequences"],
                "loop_types": data["loop_types"],
                "pair_indices": data["pair_indices"],
                "ids": data["ids"],
            }
            if data["targets"] is not None:
                save_dict["targets"] = data["targets"]

            np.savez_compressed(cache_path, **save_dict)

    # Determine augmentation
    # Augment only on training set
    augment = mode == "train"

    dataset = RNADataset(
        sequences=data["sequences"],
        loop_types=data["loop_types"],
        pair_indices=data["pair_indices"],
        targets=data["targets"],
        ids=data["ids"],
        augment=augment,
    )

    return dataset
