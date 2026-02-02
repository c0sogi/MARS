import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.utils import parse_list_column

# =========================================================================
# Mappings
# =========================================================================
TOKEN2INT_SEQ = {x: i for i, x in enumerate("AGCU")}
TOKEN2INT_STRUCT = {x: i for i, x in enumerate("().")}
TOKEN2INT_LOOP = {x: i for i, x in enumerate("SMIBHEX")}


def get_partner_indices(structure):
    """
    Parses dot-bracket structure to find partner indices.
    Returns an array of shape (L,) where arr[i] is the index of the base paired with i.
    Unpaired bases are -1.
    """
    length = len(structure)
    partner_indices = np.full(length, -1, dtype=np.int32)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                start_idx = stack.pop()
                partner_indices[start_idx] = i
                partner_indices[i] = start_idx

    return partner_indices


def one_hot_encode(seq, mapping, num_classes):
    """
    One-hot encodes a sequence string based on a mapping.
    Returns array of shape (L, num_classes).
    """
    arr = np.zeros((len(seq), num_classes), dtype=np.float32)
    for i, char in enumerate(seq):
        if char in mapping:
            arr[i, mapping[char]] = 1.0
    return arr


def process_data(mode, load_cached_data=True):
    """
    Loads and processes data, using caching to speed up subsequent runs.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Dictionary containing numpy arrays of processed data.
    """
    # Determine paths based on mode
    if mode == "train":
        csv_path = Config.TRAIN_METADATA
        cache_path = Config.TRAIN_CACHE
    elif mode == "val":
        csv_path = Config.VAL_METADATA
        cache_path = Config.VAL_CACHE
    elif mode == "test":
        csv_path = Config.TEST_METADATA
        cache_path = Config.TEST_CACHE
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Ensure working directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {mode} data from cache: {cache_path}")
        return dict(np.load(cache_path, allow_pickle=True))

    print(f"Processing {mode} data from scratch...")

    # Load Metadata
    df = pd.read_csv(csv_path)

    # Pre-allocate lists
    all_inputs = []
    all_partner_indices = []
    all_targets = []
    all_ids = df["id"].values

    # Process each sample
    for idx, row in df.iterrows():
        sequence = row["sequence"]
        structure = row["structure"]
        loop_type = row["predicted_loop_type"]
        seq_len = len(sequence)

        # 1. Basic One-Hot Features
        # Sequence: 4 channels
        ohe_seq = one_hot_encode(sequence, TOKEN2INT_SEQ, 4)
        # Structure: 3 channels
        ohe_struct = one_hot_encode(structure, TOKEN2INT_STRUCT, 3)
        # Loop Type: 7 channels
        ohe_loop = one_hot_encode(loop_type, TOKEN2INT_LOOP, 7)

        # 2. Partner Indices
        p_indices = get_partner_indices(structure)

        # 3. Partner Identity (Explicit Partner Injection)
        # Shape (L, 4). If p_indices[i] != -1, take ohe_seq[p_indices[i]], else 0.
        partner_identity = np.zeros((seq_len, 4), dtype=np.float32)
        # Mask for paired bases
        paired_mask = p_indices != -1
        # Gather partner features
        if np.any(paired_mask):
            partner_identity[paired_mask] = ohe_seq[p_indices[paired_mask]]

        # 4. Concatenate Input Features
        # Total Channels: 4 (Seq) + 3 (Struct) + 7 (Loop) + 4 (PartnerID) = 18
        sample_input = np.concatenate(
            [ohe_seq, ohe_struct, ohe_loop, partner_identity], axis=1
        )
        all_inputs.append(sample_input)
        all_partner_indices.append(p_indices)

        # 5. Process Targets (only for train/val)
        if mode in ["train", "val"]:
            # Targets are stored as stringified lists in columns
            # We need to stack them: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
            # Note: Config.ALL_TARGETS defines the order
            target_arrays = []
            for t_col in Config.ALL_TARGETS:
                # Parse string to numpy array
                t_arr = parse_list_column(row[t_col])
                target_arrays.append(t_arr)

            # Stack to shape (seq_scored, 5)
            # Transpose because target_arrays is list of (68,) -> (5, 68) -> (68, 5)
            sample_targets = np.stack(target_arrays, axis=1)

            # Pad to seq_length (107) with zeros
            # sample_targets is (68, 5), we need (107, 5)
            pad_len = seq_len - sample_targets.shape[0]
            if pad_len > 0:
                padding = np.zeros((pad_len, 5), dtype=np.float32)
                sample_targets = np.concatenate([sample_targets, padding], axis=0)

            all_targets.append(sample_targets)

    # Convert lists to numpy arrays
    # Inputs: (N, 107, 18)
    inputs_np = np.array(all_inputs, dtype=np.float32)
    # Partner Indices: (N, 107)
    partner_indices_np = np.array(all_partner_indices, dtype=np.int32)

    data_dict = {
        "inputs": inputs_np,
        "partner_indices": partner_indices_np,
        "ids": all_ids,
    }

    if mode in ["train", "val"]:
        # Targets: (N, 107, 5)
        targets_np = np.array(all_targets, dtype=np.float32)
        data_dict["targets"] = targets_np

    # Save to cache
    print(f"Saving {mode} data to cache: {cache_path}")
    np.savez_compressed(cache_path, **data_dict)

    return data_dict


class RNADataset(Dataset):
    def __init__(self, data_dict, mode="train"):
        """
        Args:
            data_dict (dict): Dictionary containing processed numpy arrays.
            mode (str): 'train', 'val', or 'test'.
        """
        self.mode = mode
        self.inputs = data_dict["inputs"]
        self.partner_indices = data_dict["partner_indices"]
        self.ids = data_dict["ids"]

        if self.mode in ["train", "val"]:
            self.targets = data_dict["targets"]
        else:
            self.targets = None

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Fetch features
        # inputs: (107, 18)
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)

        # partner_indices: (107,)
        # These are needed for the dynamic gathering in the model
        p_idx = torch.tensor(self.partner_indices[idx], dtype=torch.long)

        if self.mode in ["train", "val"]:
            # targets: (107, 5)
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            return x, p_idx, y
        else:
            # For test, return IDs to help construct submission
            sample_id = self.ids[idx]
            return x, p_idx, sample_id
