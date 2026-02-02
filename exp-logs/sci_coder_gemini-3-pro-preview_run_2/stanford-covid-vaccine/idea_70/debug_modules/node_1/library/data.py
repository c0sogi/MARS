import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# =========================================================================
# Constants & Mappings
# =========================================================================
SEQ_MAP = {c: i for i, c in enumerate("AGCU")}
STRUCT_MAP = {c: i for i, c in enumerate(".()")}
LOOP_MAP = {c: i for i, c in enumerate("SMIBHEX")}

# =========================================================================
# Helper Functions
# =========================================================================


def get_structure_pairs(structure):
    """
    Parses a dot-bracket structure string to find paired indices.
    Returns an array where arr[i] = j if i is paired with j, else -1.
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


def one_hot(seq, map_dict, width):
    """
    Generates a one-hot encoding for a sequence based on a mapping dictionary.
    """
    n = len(seq)
    res = np.zeros((n, width), dtype=np.float32)
    for i, c in enumerate(seq):
        if c in map_dict:
            res[i, map_dict[c]] = 1.0
    return res


def get_features(sequence, structure, predicted_loop_type):
    """
    Generates the hybrid input features for a single RNA sample.

    Returns:
        input_tensor: (L, 18) - Concatenation of Seq, Struct, Loop, PartnerIdentity
        partner_indices: (L,) - Indices of paired bases (-1 if unpaired)
    """
    length = len(sequence)

    # 1. Basic One-Hot Encodings
    f_seq = one_hot(sequence, SEQ_MAP, 4)  # (L, 4)
    f_struct = one_hot(structure, STRUCT_MAP, 3)  # (L, 3)
    f_loop = one_hot(predicted_loop_type, LOOP_MAP, 7)  # (L, 7)

    # 2. Partner Context
    partner_indices = get_structure_pairs(structure)  # (L,)

    # 3. Explicit Partner Identity
    # Construct (L, 4) where row i is the one-hot of sequence[partner_indices[i]]
    f_partner = np.zeros((length, 4), dtype=np.float32)
    for i, j in enumerate(partner_indices):
        if j != -1:
            # Get the base at the paired position
            char_j = sequence[j]
            if char_j in SEQ_MAP:
                f_partner[i, SEQ_MAP[char_j]] = 1.0

    # 4. Concatenate to form Hybrid Input Stem
    # Shape: (L, 4 + 3 + 7 + 4) = (L, 18)
    input_tensor = np.concatenate([f_seq, f_struct, f_loop, f_partner], axis=1)

    return input_tensor, partner_indices


# =========================================================================
# Dataset Class
# =========================================================================


class RNADataset(Dataset):
    def __init__(self, inputs, partner_indices, targets=None, ids=None):
        self.inputs = inputs
        self.partner_indices = partner_indices
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # inputs: (L, 18)
        # partner_indices: (L,)
        item = {
            "inputs": torch.tensor(self.inputs[idx], dtype=torch.float32),
            "partner_indices": torch.tensor(
                self.partner_indices[idx], dtype=torch.long
            ),
        }

        if self.targets is not None:
            # targets: (L, 5)
            item["targets"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        if self.ids is not None:
            item["ids"] = self.ids[idx]

        return item


# =========================================================================
# Data Processing & Caching
# =========================================================================


def preprocess_data(df, cache_path, load_cached_data=True):
    """
    Processes the dataframe to generate model inputs and targets.
    Implements strict caching logic.
    """
    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        try:
            data = np.load(cache_path, allow_pickle=True)
            inputs = data["inputs"]
            partner_indices = data["partner_indices"]
            ids = data["ids"]
            targets = data["targets"] if "targets" in data.files else None

            # Handle None stored in npz (it comes back as a 0-d array object if saved as None)
            if targets is not None and targets.shape == ():
                targets = None

            return inputs, partner_indices, targets, ids
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    print(f"Processing data for {len(df)} samples...")

    all_inputs = []
    all_partner_indices = []
    all_targets = []
    all_ids = df["id"].values

    # Check for targets
    target_cols = Config.TARGET_COLS
    has_targets = all(col in df.columns for col in target_cols)

    # Pre-parse target columns if they exist (string -> list)
    if has_targets:
        # We work on a copy to avoid SettingWithCopy warnings on the original DF
        df_targets = df[target_cols].copy()
        for col in target_cols:
            # Only parse if it looks like a string representation of a list
            if len(df_targets) > 0 and isinstance(df_targets[col].iloc[0], str):
                df_targets[col] = df_targets[col].apply(
                    lambda x: np.array(ast.literal_eval(x), dtype=np.float32)
                )

    # Iterate and Process
    for idx, row in df.iterrows():
        # Generate Features
        inp, p_idx = get_features(
            row["sequence"], row["structure"], row["predicted_loop_type"]
        )
        all_inputs.append(inp)
        all_partner_indices.append(p_idx)

        # Process Targets
        if has_targets:
            # The targets in JSON are length `seq_scored` (68).
            # We must pad them to `seq_length` (107).
            t_arrays = []
            for col in target_cols:
                # Get the array from our pre-parsed dataframe
                val = df_targets.at[idx, col]

                # Pad
                padded = np.zeros(Config.SEQ_LENGTH, dtype=np.float32)
                if len(val) > Config.SEQ_LENGTH:
                    padded[:] = val[: Config.SEQ_LENGTH]
                else:
                    padded[: len(val)] = val
                t_arrays.append(padded)

            # Stack to (L, 5)
            sample_target = np.stack(t_arrays, axis=1)
            all_targets.append(sample_target)

    # Convert lists to numpy arrays
    all_inputs = np.array(all_inputs, dtype=np.float32)
    all_partner_indices = np.array(all_partner_indices, dtype=np.int32)

    if has_targets:
        all_targets = np.array(all_targets, dtype=np.float32)
    else:
        all_targets = None

    # Save to Cache
    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    save_dict = {
        "inputs": all_inputs,
        "partner_indices": all_partner_indices,
        "ids": all_ids,
    }
    if all_targets is not None:
        save_dict["targets"] = all_targets

    np.savez_compressed(cache_path, **save_dict)
    print(f"Saved processed data to {cache_path}")

    return all_inputs, all_partner_indices, all_targets, all_ids


def get_dataloaders(debug=False):
    """
    Main entry point to get DataLoaders.
    Handles metadata loading, debug subsetting, and caching.
    """
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Debug Mode Handling
    cache_suffix = ""
    if debug:
        print("DEBUG MODE: Using subset of data.")
        train_df = train_df.iloc[:64]
        val_df = val_df.iloc[:32]
        test_df = test_df.iloc[:32]
        cache_suffix = "_debug"

    # Define Cache Paths (with optional debug suffix)
    train_cache = Config.CACHE_TRAIN_DATA.replace(".npz", f"{cache_suffix}.npz")
    val_cache = Config.CACHE_VAL_DATA.replace(".npz", f"{cache_suffix}.npz")
    test_cache = Config.CACHE_TEST_DATA.replace(".npz", f"{cache_suffix}.npz")

    # Process Data
    train_inputs, train_pairs, train_targets, train_ids = preprocess_data(
        train_df, train_cache, load_cached_data=True
    )

    val_inputs, val_pairs, val_targets, val_ids = preprocess_data(
        val_df, val_cache, load_cached_data=True
    )

    test_inputs, test_pairs, _, test_ids = preprocess_data(
        test_df, test_cache, load_cached_data=True
    )

    # Create Datasets
    train_ds = RNADataset(train_inputs, train_pairs, train_targets, train_ids)
    val_ds = RNADataset(val_inputs, val_pairs, val_targets, val_ids)
    test_ds = RNADataset(test_inputs, test_pairs, None, test_ids)

    # Create DataLoaders
    # Pin memory for faster transfer to GPU
    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=pin,
        drop_last=True,  # Drop incomplete batch to keep statistics stable
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=pin,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=pin,
    )

    return train_loader, val_loader, test_loader
