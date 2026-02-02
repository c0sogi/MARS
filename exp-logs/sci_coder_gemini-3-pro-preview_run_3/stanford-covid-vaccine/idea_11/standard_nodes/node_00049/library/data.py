import os
import json
import hashlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import parse_structure_to_indices

# ==========================================
# Mappings
# ==========================================
SEQ_MAP = {"A": 0, "G": 1, "U": 2, "C": 3}
STRUCT_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


# ==========================================
# Helpers
# ==========================================
def get_process_hash(config):
    """
    Generates a deterministic hash based on data processing parameters.
    Ensures cache invalidation if vocab or dimensions change.
    """
    params = {
        "seq_len": config.seq_len,
        "pred_len": config.pred_len,
        "vocab_seq": config.vocab_seq,
        "vocab_struct": config.vocab_struct,
        "vocab_loop": config.vocab_loop,
        "input_channels": config.input_channels,
    }
    # Sort keys to ensure consistent ordering
    s = json.dumps(params, sort_keys=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def one_hot_encode(indices, num_classes):
    """
    One-hot encodes a list or array of indices.
    Returns array of shape (Length, Num_Classes).
    """
    return np.eye(num_classes)[indices]


# ==========================================
# Dataset Class
# ==========================================
class RNADataset(Dataset):
    def __init__(self, data_dict, is_test=False):
        """
        PyTorch Dataset for RNA data.

        Args:
            data_dict (dict): Dictionary containing processed numpy arrays.
            is_test (bool): Whether this is the test set (no targets).
        """
        self.inputs = data_dict["inputs"]  # Shape: (N, 107, 14)
        self.pair_indices = data_dict["pair_indices"]  # Shape: (N, 107)
        self.ids = data_dict["ids"]  # Shape: (N,)
        self.is_test = is_test

        if not self.is_test:
            self.targets = data_dict["targets"]  # Shape: (N, 107, 5)
            self.masks = data_dict["masks"]  # Shape: (N, 107)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Convert inputs to float32 tensors
        inputs = torch.tensor(self.inputs[idx], dtype=torch.float32)
        # Pair indices are integers (long)
        pair_index = torch.tensor(self.pair_indices[idx], dtype=torch.long)

        if self.is_test:
            return {"inputs": inputs, "pair_index": pair_index, "id": self.ids[idx]}
        else:
            targets = torch.tensor(self.targets[idx], dtype=torch.float32)
            mask = torch.tensor(self.masks[idx], dtype=torch.float32)
            return {
                "inputs": inputs,
                "pair_index": pair_index,
                "targets": targets,
                "mask": mask,
                "id": self.ids[idx],
            }


# ==========================================
# Processing Logic
# ==========================================
def preprocess_data(df, config, is_test=False):
    """
    Converts DataFrame into numpy arrays suitable for training/inference.

    Args:
        df (pd.DataFrame): Input dataframe loaded from Parquet.
        config (Config): Configuration object.
        is_test (bool): Flag to indicate test processing.

    Returns:
        dict: Dictionary of numpy arrays.
    """
    n_samples = len(df)
    seq_len = config.seq_len

    # Pre-allocate arrays
    # Input channels: Seq(4) + Struct(3) + Loop(7) = 14
    inputs = np.zeros((n_samples, seq_len, config.input_channels), dtype=np.float32)
    pair_indices = np.zeros((n_samples, seq_len), dtype=np.int32)

    if not is_test:
        # Targets: 5 columns. Padded to seq_len.
        targets = np.zeros((n_samples, seq_len, config.num_targets), dtype=np.float32)
        masks = np.zeros((n_samples, seq_len), dtype=np.float32)

    ids = df["id"].values.astype(str)

    # Iterate over rows
    # Using itertuples for speed
    for i, row in enumerate(df.itertuples()):
        # 1. Sequence Encoding
        # Map chars to ints, default to 0 (A) if unexpected char found
        seq_ints = [SEQ_MAP.get(c, 0) for c in row.sequence]
        seq_oh = one_hot_encode(seq_ints, config.vocab_seq)

        # 2. Structure Encoding
        struct_ints = [STRUCT_MAP.get(c, 2) for c in row.structure]
        struct_oh = one_hot_encode(struct_ints, config.vocab_struct)

        # 3. Loop Type Encoding
        loop_ints = [LOOP_MAP.get(c, 5) for c in row.predicted_loop_type]
        loop_oh = one_hot_encode(loop_ints, config.vocab_loop)

        # Concatenate features along channel dimension
        # (107, 4) + (107, 3) + (107, 7) -> (107, 14)
        inputs[i] = np.concatenate([seq_oh, struct_oh, loop_oh], axis=1)

        # 4. Pair Indices (for Latent Spatial Mixing)
        # Returns array where arr[k] is index of partner, or -1 if unpaired
        pair_indices[i] = parse_structure_to_indices(row.structure)

        # 5. Targets (Train/Val only)
        if not is_test:
            scored_len = row.seq_scored

            # Retrieve target lists (preserved by Parquet)
            t_react = row.reactivity
            t_mg_ph10 = row.deg_Mg_pH10
            t_ph10 = row.deg_pH10
            t_mg_50c = row.deg_Mg_50C
            t_50c = row.deg_50C

            # Stack into (scored_len, 5) matrix
            # Note: Assuming these are lists/arrays of floats
            row_targets = np.stack(
                [t_react, t_mg_ph10, t_ph10, t_mg_50c, t_50c], axis=1
            )

            # Assign to padded array
            targets[i, :scored_len, :] = row_targets

            # Create mask (1 for scored positions, 0 for padding)
            masks[i, :scored_len] = 1.0

    result = {"inputs": inputs, "pair_indices": pair_indices, "ids": ids}

    if not is_test:
        result["targets"] = targets
        result["masks"] = masks

    return result


# ==========================================
# Main Interface
# ==========================================
def get_dataloaders(config, load_cached_data=True):
    """
    Main entry point to retrieve DataLoaders.
    Handles caching of preprocessed numpy arrays.

    Args:
        config (Config): Configuration object.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Ensure working directory exists
    os.makedirs(config.working_dir, exist_ok=True)

    # Generate unique hash for this data configuration
    process_hash = get_process_hash(config)

    loaders = []
    splits = ["train", "val", "test"]
    file_paths = [config.train_path, config.val_path, config.test_path]

    for split, path in zip(splits, file_paths):
        is_test = split == "test"
        cache_filename = f"{split}_data_{process_hash}.npz"
        cache_path = os.path.join(config.working_dir, cache_filename)

        data_dict = None

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                print(f"[{split}] Loading cached data from {cache_path}...")
                loaded = np.load(cache_path, allow_pickle=True)
                # Convert NpzFile to dict
                data_dict = {k: loaded[k] for k in loaded.files}
            except Exception as e:
                print(f"[{split}] Cache load failed ({e}). Re-processing...")

        # 2. Process from scratch if needed
        if data_dict is None:
            print(f"[{split}] Processing raw data from {path}...")
            df = pd.read_parquet(path)

            # Debugging: subset data if configured
            if config.debug and not is_test:
                print(f"[{split}] Debug mode: using first 100 samples.")
                df = df.iloc[:100]

            data_dict = preprocess_data(df, config, is_test=is_test)

            # Save to cache
            print(f"[{split}] Saving processed data to {cache_path}...")
            np.savez_compressed(cache_path, **data_dict)

        # 3. Create Dataset
        dataset = RNADataset(data_dict, is_test=is_test)

        # 4. Create DataLoader
        shuffle = split == "train"
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=shuffle,  # Drop last incomplete batch only for training
        )
        loaders.append(loader)

    return tuple(loaders)
