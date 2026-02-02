import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Constants for mappings
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {".": 0, "(": 1, ")": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


class RNADataset(Dataset):
    def __init__(self, inputs, partner_indices, targets=None, ids=None):
        """
        Args:
            inputs (np.ndarray): Shape (N, Seq_Len, Channels).
                                 Channels = Seq(4) + Struct(3) + Loop(7) + PartnerID(4).
            partner_indices (np.ndarray): Shape (N, Seq_Len). Indices of paired bases.
            targets (np.ndarray, optional): Shape (N, Seq_Len, 5).
            ids (list, optional): List of sample IDs.
        """
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.partner_indices = torch.tensor(partner_indices, dtype=torch.long)
        self.targets = (
            torch.tensor(targets, dtype=torch.float32) if targets is not None else None
        )
        self.ids = ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        sample = {
            "inputs": self.inputs[idx],
            "partner_indices": self.partner_indices[idx],
        }
        if self.targets is not None:
            sample["targets"] = self.targets[idx]
        if self.ids is not None:
            sample["id"] = self.ids[idx]
        return sample


def get_structure_map(structure):
    """
    Parses dot-bracket structure to find pairing partners.
    Returns an array where arr[i] is the index of the partner of i, or -1 if unpaired.
    """
    length = len(structure)
    partner_map = np.full(length, -1, dtype=int)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                partner_map[i] = j
                partner_map[j] = i
    return partner_map


def one_hot(seq, map_dict, num_classes):
    """
    One-hot encodes a sequence string based on a mapping dictionary.
    """
    arr = np.array([map_dict.get(c, 0) for c in seq])
    return np.eye(num_classes)[arr]


def process_dataframe(df, mode="train"):
    """
    Processes a dataframe into numpy arrays for features and targets.
    """
    num_samples = len(df)
    seq_len = 107

    # Feature Dimensions
    dim_seq = 4
    dim_struct = 3
    dim_loop = 7
    dim_partner = 4
    total_dim = dim_seq + dim_struct + dim_loop + dim_partner

    # Initialize arrays
    inputs = np.zeros((num_samples, seq_len, total_dim), dtype=np.float32)
    partner_indices = np.full((num_samples, seq_len), -1, dtype=np.int32)

    # Target columns
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    targets = (
        np.zeros((num_samples, seq_len, 5), dtype=np.float32)
        if mode != "test"
        else None
    )

    ids = df["id"].tolist()

    for idx, row in df.iterrows():
        # 1. Base Features
        seq_oh = one_hot(row["sequence"], SEQ_MAP, 4)
        struct_oh = one_hot(row["structure"], STRUCT_MAP, 3)
        loop_oh = one_hot(row["predicted_loop_type"], LOOP_MAP, 7)

        # 2. Partner Mapping
        p_map = get_structure_map(row["structure"])
        partner_indices[idx] = p_map

        # 3. Partner Identity Feature
        # If p_map[i] != -1, partner_id[i] = seq_oh[p_map[i]]
        # Else partner_id[i] = 0
        partner_id_oh = np.zeros((seq_len, 4), dtype=np.float32)
        valid_pairs = p_map != -1
        # Use integer indexing for speed
        # Convert sequence string to indices first
        seq_indices = np.array([SEQ_MAP.get(c, 0) for c in row["sequence"]])

        # Where valid pairs exist, get the index of the partner base
        partner_base_indices = seq_indices[p_map[valid_pairs]]
        # Set the one-hot
        partner_id_oh[valid_pairs, partner_base_indices] = 1.0

        # Concatenate all features
        inputs[idx] = np.concatenate(
            [seq_oh, struct_oh, loop_oh, partner_id_oh], axis=1
        )

        # 4. Targets (Train/Val only)
        if mode != "test":
            for t_i, col in enumerate(target_cols):
                # Parse stringified list
                val_str = row[col]
                try:
                    val_list = ast.literal_eval(val_str)
                except:
                    val_list = []

                # Pad to seq_len (107). The provided lists are usually length 68.
                # We fill the rest with 0.
                length_provided = len(val_list)
                if length_provided > 0:
                    targets[idx, :length_provided, t_i] = val_list

    return inputs, partner_indices, targets, ids


def get_data(mode, load_cached_data=True):
    """
    Loads data from cache or processes from scratch.
    """
    cache_dir = "./working/idea_45/"
    os.makedirs(cache_dir, exist_ok=True)

    cache_file = os.path.join(cache_dir, f"{mode}_data_rdf_rn_v1.npz")

    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading {mode} data from cache: {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        inputs = data["inputs"]
        partner_indices = data["partner_indices"]
        ids = data["ids"].tolist()
        targets = data["targets"] if "targets" in data else None
        return inputs, partner_indices, targets, ids

    print(f"Processing {mode} data from scratch...")

    # Load metadata CSV
    if mode == "train":
        df = pd.read_csv("./metadata/train.csv")
    elif mode == "val":
        df = pd.read_csv("./metadata/val.csv")
    elif mode == "test":
        df = pd.read_csv("./metadata/test.csv")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    inputs, partner_indices, targets, ids = process_dataframe(df, mode)

    # Save to cache
    save_dict = {
        "inputs": inputs,
        "partner_indices": partner_indices,
        "ids": np.array(ids),
    }
    if targets is not None:
        save_dict["targets"] = targets

    np.savez_compressed(cache_file, **save_dict)
    print(f"Saved {mode} data to cache: {cache_file}")

    return inputs, partner_indices, targets, ids


def get_dataloaders(batch_size=32, num_workers=2, load_cached_data=True):
    """
    Creates DataLoaders for train, val, and test sets.

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.
        load_cached_data (bool): Whether to use cached .npz files.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Load Data
    train_inputs, train_pi, train_targets, train_ids = get_data(
        "train", load_cached_data
    )
    val_inputs, val_pi, val_targets, val_ids = get_data("val", load_cached_data)
    test_inputs, test_pi, _, test_ids = get_data("test", load_cached_data)

    # Create Datasets
    train_dataset = RNADataset(train_inputs, train_pi, train_targets, train_ids)
    val_dataset = RNADataset(val_inputs, val_pi, val_targets, val_ids)
    test_dataset = RNADataset(test_inputs, test_pi, None, test_ids)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
