import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed

# =============================================================================
# MAPPINGS
# =============================================================================
TOKEN2INT_SEQ = {x: i for i, x in enumerate("AGCU")}
TOKEN2INT_STRUCT = {x: i for i, x in enumerate("().")}
TOKEN2INT_LOOP = {x: i for i, x in enumerate("SMIBHEX")}


def get_partner_idx(structure, seq_length):
    """
    Parses a dot-bracket structure string to find the index of the paired base.
    Returns a numpy array where arr[i] is the index of the base paired with i.
    If unpaired, arr[i] is -1.
    """
    partner = np.full(seq_length, -1, dtype=np.int32)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                partner[i] = j
                partner[j] = i

    return partner


def get_one_hot(seq, mapping, length, num_classes):
    """
    Converts a sequence string into a one-hot encoded numpy array.
    """
    arr = np.zeros((length, num_classes), dtype=np.float32)
    for i, char in enumerate(seq):
        if i >= length:
            break
        if char in mapping:
            arr[i, mapping[char]] = 1.0
    return arr


def parse_list_col(x):
    """Parses a stringified list from CSV into a list of floats."""
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return []


def process_data(csv_path, mode, config, load_cached_data=True):
    """
    Loads data from CSV, performs feature engineering, and caches the result.

    Args:
        csv_path (str): Path to the metadata CSV file.
        mode (str): 'train', 'val', or 'test'.
        config (Config): Configuration object.
        load_cached_data (bool): Whether to try loading from cache.

    Returns:
        tuple: (inputs, partner_indices, targets, ids)
    """
    cache_file = f"{mode}_data_sf_dcn_v1.npz"
    cache_path = os.path.join(config.working_dir, cache_file)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {mode} data from cache: {cache_path}")
        try:
            data = np.load(cache_path, allow_pickle=True)
            return (
                data["inputs"],
                data["partner_indices"],
                data["targets"],
                data["ids"],
            )
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from scratch
    print(f"Processing {mode} data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Filter for debugging if needed (not implemented here to keep full dataset)
    # if config.debug:
    #     df = df.head(100)

    num_samples = len(df)
    seq_len = config.seq_length
    input_channels = config.input_channels

    # Initialize arrays
    # Inputs: (N, Channels, Seq_Len) - Transposed later if needed, constructing as (N, Seq_Len, Channels) first
    # Actually, let's construct as (N, Seq_Len, Channels) then transpose to (N, Channels, Seq_Len)
    # Channels: Seq(4) + Struct(3) + Loop(7) + PartnerID(4) = 18
    inputs = np.zeros((num_samples, seq_len, input_channels), dtype=np.float32)
    partner_indices = np.zeros((num_samples, seq_len), dtype=np.int32)

    # Targets: (N, Seq_Len, 5)
    # If test mode, we fill with zeros
    targets = np.zeros((num_samples, seq_len, 5), dtype=np.float32)

    ids = df["id"].values

    # Pre-compute target column parsing if not test
    if mode != "test":
        target_cols = (
            config.target_cols
        )  # ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
        # Parse columns once
        parsed_targets = {}
        for col in target_cols:
            parsed_targets[col] = df[col].apply(parse_list_col).values

    for idx, row in df.iterrows():
        # --- Feature Engineering ---
        sequence = row["sequence"]
        structure = row["structure"]
        loop_type = row["predicted_loop_type"]

        # 1. Sequence One-Hot (4)
        seq_oh = get_one_hot(sequence, TOKEN2INT_SEQ, seq_len, 4)

        # 2. Structure One-Hot (3)
        struct_oh = get_one_hot(structure, TOKEN2INT_STRUCT, seq_len, 3)

        # 3. Loop Type One-Hot (7)
        loop_oh = get_one_hot(loop_type, TOKEN2INT_LOOP, seq_len, 7)

        # 4. Partner Identity (4)
        # First get indices
        pidx = get_partner_idx(structure, seq_len)
        partner_indices[idx] = pidx

        # Construct Partner Identity Feature
        # If pidx[i] == -1, vector is 0. Else it is one-hot of sequence[pidx[i]]
        partner_oh = np.zeros((seq_len, 4), dtype=np.float32)
        for i, p_i in enumerate(pidx):
            if p_i != -1 and p_i < seq_len:
                base = sequence[p_i]
                if base in TOKEN2INT_SEQ:
                    partner_oh[i, TOKEN2INT_SEQ[base]] = 1.0

        # Concatenate features along channel dimension
        # Shape: (Seq_Len, 18)
        sample_feat = np.concatenate([seq_oh, struct_oh, loop_oh, partner_oh], axis=1)
        inputs[idx] = sample_feat

        # --- Targets ---
        if mode != "test":
            for t_i, col in enumerate(config.target_cols):
                val_list = parsed_targets[col][idx]
                # Pad or truncate to seq_len (though data should be 68 long usually, we map to 107 space)
                # The task description says targets are length seq_scored (68).
                # We place them in the first 68 positions of the (107,) array.
                length = len(val_list)
                if length > 0:
                    # Ensure we don't exceed seq_len
                    fill_len = min(length, seq_len)
                    targets[idx, :fill_len, t_i] = val_list[:fill_len]

    # Transpose inputs to (N, Channels, Seq_Len) for PyTorch Conv1d
    inputs = inputs.transpose(0, 2, 1)

    # 3. Save to cache
    os.makedirs(config.working_dir, exist_ok=True)
    print(f"Saving {mode} data to cache: {cache_path}")
    np.savez_compressed(
        cache_path,
        inputs=inputs,
        partner_indices=partner_indices,
        targets=targets,
        ids=ids,
    )

    return inputs, partner_indices, targets, ids


class RNADataset(Dataset):
    def __init__(self, inputs, partner_indices, targets, ids):
        self.inputs = inputs
        self.partner_indices = partner_indices
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Inputs: (Channels, Seq_Len)
        x = torch.from_numpy(self.inputs[idx])

        # Partner Indices: (Seq_Len,)
        # We keep -1 for unpaired. The model handles masking/gathering logic.
        pidx = torch.from_numpy(self.partner_indices[idx]).long()

        # Targets: (Seq_Len, 5)
        y = torch.from_numpy(self.targets[idx])

        # ID
        id_val = self.ids[idx]

        return x, pidx, y, id_val


def get_dataloaders(config, load_cached_data=True):
    """
    Generates DataLoaders for train, val, and test sets.

    Args:
        config (Config): Configuration object.
        load_cached_data (bool): Whether to use cached .npz files.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Process/Load Data
    train_inputs, train_pidx, train_targets, _ = process_data(
        config.train_csv, "train", config, load_cached_data
    )
    val_inputs, val_pidx, val_targets, _ = process_data(
        config.val_csv, "val", config, load_cached_data
    )
    test_inputs, test_pidx, test_targets, test_ids = process_data(
        config.test_csv, "test", config, load_cached_data
    )

    # Create Datasets
    train_dataset = RNADataset(train_inputs, train_pidx, train_targets, _)
    val_dataset = RNADataset(val_inputs, val_pidx, val_targets, _)
    test_dataset = RNADataset(test_inputs, test_pidx, test_targets, test_ids)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
