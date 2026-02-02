import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config


def one_hot_encode(seq, alphabet):
    """
    One-hot encodes a sequence string based on the provided alphabet.

    Args:
        seq (str): Input sequence.
        alphabet (str): String of all possible characters.

    Returns:
        np.ndarray: One-hot encoded array of shape (len(seq), len(alphabet)).
    """
    char_to_idx = {char: idx for idx, char in enumerate(alphabet)}
    seq_len = len(seq)
    vocab_size = len(alphabet)
    one_hot = np.zeros((seq_len, vocab_size), dtype=np.float32)

    for i, char in enumerate(seq):
        if char in char_to_idx:
            one_hot[i, char_to_idx[char]] = 1.0

    return one_hot


def get_adjacency_info(structure):
    """
    Parses a dot-bracket structure string to generate adjacency information
    for the Decoupled Structural Interaction Module.

    Args:
        structure (str): Dot-bracket notation string (e.g., ".(())").

    Returns:
        tuple:
            indices (np.ndarray): Array where indices[i] = j if i is paired with j.
                                  If unpaired, indices[i] = i (self-loop).
            mask (np.ndarray): Binary mask where 1.0 indicates a paired base, 0.0 unpaired.
    """
    seq_len = len(structure)
    indices = np.arange(seq_len)
    mask = np.zeros(seq_len, dtype=np.float32)

    stack = []
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                # Bidirectional linkage
                indices[i] = j
                indices[j] = i
                # Mask as paired
                mask[i] = 1.0
                mask[j] = 1.0

    return indices, mask


def process_dataframe(df, config, is_test=False):
    """
    Converts a pandas DataFrame into a dictionary of numpy arrays.

    Args:
        df (pd.DataFrame): Input dataframe.
        config (Config): Configuration object.
        is_test (bool): Whether this is the test set (no targets).

    Returns:
        dict: Dictionary containing 'inputs', 'bpp_indices', 'bpp_masks', 'targets', 'ids'.
    """
    # Alphabets for feature channels
    # 1. Sequence (4)
    seq_alphabet = "AGUC"
    # 2. Structure (3) - Using standard dot-bracket chars
    struct_alphabet = ".()"
    # 3. Loop Type (7)
    loop_alphabet = "SMIBHEX"

    num_samples = len(df)
    seq_len = config.seq_len

    # Pre-allocate arrays
    # Total features = 4 + 3 + 7 = 14
    inputs = np.zeros((num_samples, seq_len, 14), dtype=np.float32)
    bpp_indices = np.zeros((num_samples, seq_len), dtype=np.int64)
    bpp_masks = np.zeros((num_samples, seq_len), dtype=np.float32)
    ids = df["id"].values

    # Pre-allocate targets if not test
    if not is_test:
        targets = np.zeros(
            (num_samples, config.pred_len, config.num_targets), dtype=np.float32
        )
    else:
        targets = None

    for idx, row in df.iterrows():
        # --- Feature Encoding ---
        seq_oh = one_hot_encode(row["sequence"], seq_alphabet)
        struct_oh = one_hot_encode(row["structure"], struct_alphabet)
        loop_oh = one_hot_encode(row["predicted_loop_type"], loop_alphabet)

        # Concatenate channel-wise
        inputs[idx] = np.concatenate([seq_oh, struct_oh, loop_oh], axis=1)

        # --- Structural Adjacency ---
        s_indices, s_mask = get_adjacency_info(row["structure"])
        bpp_indices[idx] = s_indices
        bpp_masks[idx] = s_mask

        # --- Targets ---
        if not is_test:
            # target_cols order: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
            for t_i, col in enumerate(config.target_cols):
                val = row[col]
                # Parquet usually preserves lists/arrays.
                # Ensure we take the first 'pred_len' (68) elements.
                if isinstance(val, (list, np.ndarray)):
                    targets[idx, :, t_i] = val[: config.pred_len]
                else:
                    # Should not happen given metadata verification, but safe fallback
                    pass

    return {
        "inputs": inputs,
        "bpp_indices": bpp_indices,
        "bpp_masks": bpp_masks,
        "targets": targets,
        "ids": ids,
    }


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA degradation prediction.
    """

    def __init__(self, data_dict, is_test=False):
        self.inputs = torch.from_numpy(data_dict["inputs"]).float()
        self.bpp_indices = torch.from_numpy(data_dict["bpp_indices"]).long()
        # Unsqueeze mask to (N, L, 1) for broadcasting in model
        self.bpp_masks = torch.from_numpy(data_dict["bpp_masks"]).float().unsqueeze(-1)
        self.ids = data_dict["ids"]
        self.is_test = is_test

        if not is_test and data_dict["targets"] is not None:
            self.targets = torch.from_numpy(data_dict["targets"]).float()
        else:
            self.targets = None

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        sample = {
            "sequence": self.inputs[idx],  # Shape: (107, 14)
            "bpp_indices": self.bpp_indices[idx],  # Shape: (107,)
            "bpp_mask": self.bpp_masks[idx],  # Shape: (107, 1)
            "id": self.ids[idx],
        }

        if not self.is_test and self.targets is not None:
            sample["targets"] = self.targets[idx]  # Shape: (68, 5)

        return sample


def load_or_process_data(split, config, load_cached_data=True):
    """
    Loads data from cache if available and requested.
    Otherwise, processes from raw parquet files and saves to cache.

    Uses .npz format to store multiple arrays efficiently without pickle issues.

    Args:
        split (str): 'train', 'val', or 'test'.
        config (Config): Configuration object.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        RNADataset: The instantiated dataset object.
    """
    # Determine raw and cache paths
    if split == "train":
        raw_path = config.train_data_path
        # Modify extension to .npz for multi-array storage
        cache_path = config.train_cache_path.replace(".npy", ".npz")
    elif split == "val":
        raw_path = config.val_data_path
        cache_path = config.val_cache_path.replace(".npy", ".npz")
    elif split == "test":
        raw_path = config.test_data_path
        cache_path = config.test_cache_path.replace(".npy", ".npz")
    else:
        raise ValueError(f"Unknown split: {split}")

    # Ensure cache directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    data_dict = None

    # 1. Attempt to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            # Load .npz file
            loaded = np.load(cache_path)
            data_dict = {
                "inputs": loaded["inputs"],
                "bpp_indices": loaded["bpp_indices"],
                "bpp_masks": loaded["bpp_masks"],
                "ids": loaded["ids"],
            }

            # Load targets if they exist and we are not in test mode
            if "targets" in loaded and split != "test":
                data_dict["targets"] = loaded["targets"]
            else:
                data_dict["targets"] = None

        except Exception:
            # If loading fails, fall back to processing
            data_dict = None

    # 2. Process from raw data if cache miss or load failed
    if data_dict is None:
        # Load Parquet file
        df = pd.read_parquet(raw_path)

        # Handle Debug Mode
        if config.debug:
            df = df.head(config.debug_subset_size)

        is_test = split == "test"
        data_dict = process_dataframe(df, config, is_test=is_test)

        # Prepare dict for saving
        save_dict = {
            "inputs": data_dict["inputs"],
            "bpp_indices": data_dict["bpp_indices"],
            "bpp_masks": data_dict["bpp_masks"],
            "ids": data_dict["ids"],
        }
        if not is_test and data_dict["targets"] is not None:
            save_dict["targets"] = data_dict["targets"]

        # Save compressed .npz
        np.savez(cache_path, **save_dict)

    return RNADataset(data_dict, is_test=(split == "test"))
