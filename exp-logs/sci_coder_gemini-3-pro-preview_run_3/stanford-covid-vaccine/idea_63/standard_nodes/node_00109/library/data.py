import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# ==========================================
# Constants and Mappings
# ==========================================
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

# ==========================================
# Helper Functions
# ==========================================


def get_pair_indices(structure):
    """
    Parses a dot-bracket structure string to find pairing partners.

    Args:
        structure (str): Dot-bracket string (e.g., "((..))").

    Returns:
        indices (np.ndarray): Shape (seq_len,). indices[i] is the index of the partner of i.
                              If unpaired, indices[i] = 0 (safe index for gather operations).
        mask (np.ndarray): Shape (seq_len,). mask[i] = 1.0 if paired, 0.0 if unpaired.
    """
    n = len(structure)
    indices = np.zeros(n, dtype=np.int64)
    mask = np.zeros(n, dtype=np.float32)

    stack = []
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                indices[i] = j
                indices[j] = i
                mask[i] = 1.0
                mask[j] = 1.0

    return indices, mask


def one_hot_encode(seq, mapping, num_classes):
    """
    One-hot encodes a sequence string based on a provided mapping.
    """
    arr = np.zeros((len(seq), num_classes), dtype=np.float32)
    for i, char in enumerate(seq):
        if char in mapping:
            arr[i, mapping[char]] = 1.0
    return arr


def process_dataframe(df, is_test=False):
    """
    Converts a pandas DataFrame into numpy arrays suitable for the model.

    Args:
        df (pd.DataFrame): Input dataframe.
        is_test (bool): Whether processing test data (no targets).

    Returns:
        tuple: (inputs, pair_indices, pair_masks, targets, ids)
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LEN
    input_channels = Config.INPUT_CHANNELS  # 14

    # Pre-allocate arrays for efficiency
    inputs = np.zeros((num_samples, seq_len, input_channels), dtype=np.float32)
    pair_indices = np.zeros((num_samples, seq_len), dtype=np.int64)
    pair_masks = np.zeros((num_samples, seq_len), dtype=np.float32)

    if not is_test:
        targets = np.zeros((num_samples, seq_len, Config.NUM_TARGETS), dtype=np.float32)
    else:
        targets = None

    ids = df["id"].values

    for i, row in df.iterrows():
        # 1. Sequence Features (4 channels)
        seq_oh = one_hot_encode(row["sequence"], SEQ_MAP, 4)

        # 2. Structure Features (3 channels)
        struct_oh = one_hot_encode(row["structure"], STRUCT_MAP, 3)

        # 3. Loop Type Features (7 channels)
        loop_oh = one_hot_encode(row["predicted_loop_type"], LOOP_MAP, 7)

        # Concatenate all features: Shape (107, 14)
        inputs[i] = np.concatenate([seq_oh, struct_oh, loop_oh], axis=1)

        # 4. Interaction Indices and Masks
        p_idx, p_mask = get_pair_indices(row["structure"])
        pair_indices[i] = p_idx
        pair_masks[i] = p_mask

        # 5. Targets (Train/Val only)
        if not is_test:
            # Targets are provided for the first `seq_scored` positions (68).
            # We pad the rest with zeros to match seq_len (107).
            for t_idx, col in enumerate(Config.TARGET_COLS):
                val_list = row[col]
                # Ensure it's a list/array before assignment
                if isinstance(val_list, (list, np.ndarray)):
                    length = len(val_list)
                    # Assign to the first 'length' positions
                    targets[i, :length, t_idx] = val_list

    return inputs, pair_indices, pair_masks, targets, ids


# ==========================================
# Dataset Class
# ==========================================


class RNADataset(Dataset):
    def __init__(self, inputs, pair_indices, pair_masks, targets=None, ids=None):
        self.inputs = inputs
        self.pair_indices = pair_indices
        self.pair_masks = pair_masks
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Input features: (107, 14)
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)

        # Pair indices for gather: (107,)
        p_idx = torch.tensor(self.pair_indices[idx], dtype=torch.long)

        # Pair mask for zeroing unpaired interactions: (107,)
        p_mask = torch.tensor(self.pair_masks[idx], dtype=torch.float32)

        sample = {"inputs": x, "pair_indices": p_idx, "pair_mask": p_mask}

        if self.targets is not None:
            # Targets: (107, 5) - padded with zeros
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            sample["targets"] = y

        if self.ids is not None:
            sample["id"] = self.ids[idx]

        return sample


# ==========================================
# Main Data Loading Function
# ==========================================


def get_dataloaders(load_cached_data=True, debug=False):
    """
    Loads data, handles caching, and returns PyTorch DataLoaders.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed .npz files.
        debug (bool): If True, loads a small subset of training data for testing.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file paths
    train_cache = os.path.join(cache_dir, "train_cache.npz")
    val_cache = os.path.join(cache_dir, "val_cache.npz")
    test_cache = os.path.join(cache_dir, "test_cache.npz")

    def load_or_process(mode, cache_path, metadata_path):
        """Helper to load from cache or process from metadata."""
        is_test = mode == "test"

        # Attempt to load cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading {mode} data from cache: {cache_path}")
            try:
                data = np.load(cache_path, allow_pickle=True)
                inputs = data["inputs"]
                pair_indices = data["pair_indices"]
                pair_masks = data["pair_masks"]
                ids = data["ids"]

                # Handle targets (None for test)
                if "targets" in data:
                    targets = data["targets"]
                    # np.savez might save None as a 0-d array object
                    if targets.shape == ():
                        targets = None
                else:
                    targets = None

                return inputs, pair_indices, pair_masks, targets, ids
            except Exception as e:
                print(f"Failed to load cache for {mode}: {e}. Reprocessing...")

        # Process from scratch
        print(f"Processing {mode} data from metadata: {metadata_path}")
        df = pd.read_parquet(metadata_path)

        if debug and mode == "train":
            print(
                f"DEBUG: Using subset of {Config.DEBUG_SUBSET_SIZE} samples for training."
            )
            df = df.iloc[: Config.DEBUG_SUBSET_SIZE]

        inputs, pair_indices, pair_masks, targets, ids = process_dataframe(
            df, is_test=is_test
        )

        # Save to cache
        save_dict = {
            "inputs": inputs,
            "pair_indices": pair_indices,
            "pair_masks": pair_masks,
            "ids": ids,
        }
        if targets is not None:
            save_dict["targets"] = targets

        np.savez(cache_path, **save_dict)
        print(f"Saved {mode} data to cache: {cache_path}")

        return inputs, pair_indices, pair_masks, targets, ids

    # 1. Load Data
    train_inputs, train_pidx, train_pmask, train_targets, train_ids = load_or_process(
        "train", train_cache, Config.TRAIN_METADATA
    )

    val_inputs, val_pidx, val_pmask, val_targets, val_ids = load_or_process(
        "val", val_cache, Config.VAL_METADATA
    )

    test_inputs, test_pidx, test_pmask, _, test_ids = load_or_process(
        "test", test_cache, Config.TEST_METADATA
    )

    # 2. Create Datasets
    train_dataset = RNADataset(
        train_inputs, train_pidx, train_pmask, train_targets, train_ids
    )
    val_dataset = RNADataset(val_inputs, val_pidx, val_pmask, val_targets, val_ids)
    test_dataset = RNADataset(test_inputs, test_pidx, test_pmask, None, test_ids)

    # 3. Create DataLoaders
    # Drop last for train to ensure consistent batch statistics if needed, though LayerNorm handles variable sizes well.
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
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
