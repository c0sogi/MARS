import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# ==========================================
# Constants & Mappings
# ==========================================
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def get_structure_info(structure_str, seq_len):
    """
    Parses a dot-bracket structure string to generate pair indices and masks.

    Args:
        structure_str (str): Dot-bracket string (e.g., "((..))").
        seq_len (int): Length of the sequence.

    Returns:
        pair_indices (np.ndarray): Array of shape (L,) where index i contains the index of its pair.
                                   If unpaired, contains i (self-loop).
        pair_mask (np.ndarray): Array of shape (L, 1). 1.0 if paired, 0.0 if unpaired.
    """
    pair_indices = np.arange(seq_len, dtype=np.int64)
    pair_mask = np.zeros((seq_len, 1), dtype=np.float32)

    stack = []
    for i, char in enumerate(structure_str):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                start = stack.pop()
                pair_indices[start] = i
                pair_indices[i] = start
                pair_mask[start] = 1.0
                pair_mask[i] = 1.0

    return pair_indices, pair_mask


def one_hot_encode(seq, mapping, vocab_size):
    """
    One-hot encodes a sequence string based on a mapping.
    """
    indices = [mapping.get(char, 0) for char in seq]
    return np.eye(vocab_size)[indices]


def process_dataframe(df, is_test=False):
    """
    Processes a pandas DataFrame into numpy arrays for the model.
    """
    num_samples = len(df)
    seq_len = Config.seq_len

    # Initialize containers
    # Features: (N, L, 14) -> 4 (Seq) + 3 (Struct) + 7 (Loop)
    features = np.zeros((num_samples, seq_len, Config.input_dim), dtype=np.float32)
    pair_indices = np.zeros((num_samples, seq_len), dtype=np.int64)
    pair_masks = np.zeros((num_samples, seq_len, 1), dtype=np.float32)
    ids = df["id"].values

    # Targets: (N, 68, 5)
    # If test, we create dummy targets
    target_len = Config.pred_len
    num_targets = Config.num_classes
    targets = np.zeros((num_samples, target_len, num_targets), dtype=np.float32)

    # Process rows
    for i, row in enumerate(df.itertuples()):
        # 1. Features
        seq_oh = one_hot_encode(row.sequence, SEQ_MAP, 4)
        struct_oh = one_hot_encode(row.structure, STRUCT_MAP, 3)
        loop_oh = one_hot_encode(row.predicted_loop_type, LOOP_MAP, 7)

        features[i] = np.concatenate([seq_oh, struct_oh, loop_oh], axis=1)

        # 2. Topology
        p_idx, p_mask = get_structure_info(row.structure, seq_len)
        pair_indices[i] = p_idx
        pair_masks[i] = p_mask

        # 3. Targets (if available)
        if not is_test:
            # Stack the 5 target columns
            # Each column in the DF is a list of floats
            t_list = []
            for col in Config.target_columns:
                val = getattr(row, col)
                # Ensure it's a list or array
                if isinstance(val, (list, np.ndarray)):
                    t_list.append(val)
                else:
                    # Fallback for scalar 0 or similar issues (though parquet usually preserves)
                    t_list.append(np.zeros(target_len))

            # Shape (5, 68) -> Transpose to (68, 5)
            t_stack = np.array(t_list, dtype=np.float32).T

            # Safety check on length
            if t_stack.shape[0] > target_len:
                t_stack = t_stack[:target_len]
            elif t_stack.shape[0] < target_len:
                # Pad if necessary (unlikely given dataset specs)
                pad = np.zeros((target_len - t_stack.shape[0], num_targets))
                t_stack = np.vstack([t_stack, pad])

            targets[i] = t_stack

    return {
        "features": features,
        "pair_indices": pair_indices,
        "pair_masks": pair_masks,
        "targets": targets,
        "ids": ids,
    }


class RNADataset(Dataset):
    def __init__(self, data_dict, is_test=False):
        self.features = data_dict["features"]
        self.pair_indices = data_dict["pair_indices"]
        self.pair_masks = data_dict["pair_masks"]
        self.targets = data_dict["targets"]
        self.ids = data_dict["ids"]
        self.is_test = is_test

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Convert to tensors
        feat = torch.from_numpy(self.features[idx])
        p_idx = torch.from_numpy(self.pair_indices[idx])
        p_mask = torch.from_numpy(self.pair_masks[idx])
        tgt = torch.from_numpy(self.targets[idx])
        sample_id = self.ids[idx]

        return feat, p_idx, p_mask, tgt, sample_id


def get_loaders(load_cached_data=True):
    """
    Prepares DataLoaders for Train, Validation, and Test sets.
    Handles caching of preprocessed numpy arrays.
    """

    # Ensure working directory exists
    os.makedirs(Config.working_dir, exist_ok=True)

    datasets = {}

    # Define tasks
    tasks = [
        ("train", Config.train_file, Config.train_cache, False),
        ("val", Config.val_file, Config.val_cache, False),
        ("test", Config.test_file, Config.test_cache, True),
    ]

    for name, file_path, cache_path, is_test in tasks:
        data_dict = None

        # 1. Try Load Cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                print(f"Loading {name} data from cache: {cache_path}")
                loaded = np.load(cache_path, allow_pickle=True)
                data_dict = {
                    "features": loaded["features"],
                    "pair_indices": loaded["pair_indices"],
                    "pair_masks": loaded["pair_masks"],
                    "targets": loaded["targets"],
                    "ids": loaded["ids"],
                }
            except Exception as e:
                print(f"Failed to load cache for {name}: {e}")
                data_dict = None

        # 2. Process if needed
        if data_dict is None:
            print(f"Processing {name} data from {file_path}...")
            df = pd.read_parquet(file_path)

            # Debugging subset
            if Config.debug:
                df = df.iloc[: Config.debug_subset_size].copy()
                print(f"Debug mode: subsetting {name} to {len(df)} samples.")

            data_dict = process_dataframe(df, is_test=is_test)

            # Save to cache
            print(f"Saving {name} data to cache: {cache_path}")
            np.savez(
                cache_path,
                features=data_dict["features"],
                pair_indices=data_dict["pair_indices"],
                pair_masks=data_dict["pair_masks"],
                targets=data_dict["targets"],
                ids=data_dict["ids"],
            )

        # 3. Create Dataset
        datasets[name] = RNADataset(data_dict, is_test=is_test)

    # 4. Create DataLoaders
    train_loader = DataLoader(
        datasets["train"],
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        datasets["val"],
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        datasets["test"],
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
