import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config

# ==================================================================================
# HELPER FUNCTIONS
# ==================================================================================


def get_structure_adj(structure):
    """
    Parses dot-bracket structure to find pairs.
    Returns a mapping array where arr[i] = j if i pairs with j, else -1.
    """
    pairs = np.full(len(structure), -1, dtype=int)
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


def get_one_hot(sequence, mapping):
    """
    Converts a sequence string into a one-hot encoded numpy array based on the provided mapping.
    """
    seq_len = len(sequence)
    num_classes = len(mapping)
    one_hot = np.zeros((seq_len, num_classes), dtype=np.float32)

    for i, char in enumerate(sequence):
        if char in mapping:
            one_hot[i, mapping[char]] = 1.0

    return one_hot


def get_partner_identity(sequence, pairs, seq_map):
    """
    Generates the one-hot encoding of the paired base for each position.
    If a base is unpaired, the vector is all zeros.
    """
    seq_len = len(sequence)
    num_bases = len(seq_map)
    partner_identity = np.zeros((seq_len, num_bases), dtype=np.float32)

    for i, pair_idx in enumerate(pairs):
        if pair_idx != -1:
            partner_char = sequence[pair_idx]
            if partner_char in seq_map:
                partner_identity[i, seq_map[partner_char]] = 1.0

    return partner_identity


# ==================================================================================
# MAIN PROCESSING FUNCTION
# ==================================================================================


def process_data(mode="train", load_cached_data=True):
    """
    Loads metadata, generates features (One-Hot + Partner Identity), and targets.
    Handles caching using .npz files.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Contains 'inputs', 'partner_indices', 'targets' (if not test), 'ids'.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Determine cache path
    cache_filename = getattr(Config, f"CACHE_{mode.upper()}")
    cache_path = os.path.join(Config.CACHE_DIR, cache_filename)

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {mode} data from {cache_path}...")
        try:
            data = np.load(cache_path, allow_pickle=True)
            result = {
                "inputs": data["inputs"],
                "partner_indices": data["partner_indices"],
                "ids": data["ids"],
            }
            if "targets" in data:
                result["targets"] = data["targets"]
            return result
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    print(f"Processing {mode} data from scratch...")

    # 2. Load Metadata
    meta_filename = getattr(Config, f"{mode.upper()}_FILE")
    meta_path = os.path.join(Config.METADATA_DIR, meta_filename)

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df = pd.read_csv(meta_path)

    # 3. Define Mappings
    seq_map = {"A": 0, "G": 1, "C": 2, "U": 3}
    struct_map = {".": 0, "(": 1, ")": 2}
    loop_map = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

    num_samples = len(df)
    seq_len = Config.SEQ_LENGTH

    # Initialize containers
    # Channels: Seq(4) + Struct(3) + Loop(7) + Partner(4) = 18
    inputs = np.zeros((num_samples, seq_len, Config.INPUT_CHANNELS), dtype=np.float32)
    partner_indices = np.full((num_samples, seq_len), -1, dtype=np.int32)
    ids = df["id"].values

    targets = None
    if mode != "test":
        targets = np.zeros((num_samples, seq_len, Config.NUM_TARGETS), dtype=np.float32)

    # 4. Feature Engineering Loop
    for idx, row in df.iterrows():
        seq = row["sequence"]
        struct = row["structure"]
        loop = row["predicted_loop_type"]

        # Generate One-Hot Features
        oh_seq = get_one_hot(seq, seq_map)  # (L, 4)
        oh_struct = get_one_hot(struct, struct_map)  # (L, 3)
        oh_loop = get_one_hot(loop, loop_map)  # (L, 7)

        # Generate Partner Info
        pairs = get_structure_adj(struct)
        oh_partner = get_partner_identity(seq, pairs, seq_map)  # (L, 4)

        # Concatenate Features
        # Order: Sequence, Structure, Loop, Partner Identity
        sample_input = np.concatenate([oh_seq, oh_struct, oh_loop, oh_partner], axis=1)
        inputs[idx] = sample_input
        partner_indices[idx] = pairs

        # Parse Targets
        if mode != "test":
            # Columns: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
            # We assume the order in Config.SCORED_TARGETS + others matches the order we want in the tensor
            # Typically: [reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C]
            # We will parse all 5.
            target_cols = [
                "reactivity",
                "deg_Mg_pH10",
                "deg_pH10",
                "deg_Mg_50C",
                "deg_50C",
            ]

            for t_i, col in enumerate(target_cols):
                val_str = row[col]
                try:
                    val_list = ast.literal_eval(val_str)
                except (ValueError, SyntaxError):
                    # Fallback for potential malformed strings or empty
                    val_list = []

                # Targets are usually length 68 (seq_scored), need to pad to 107 or place correctly
                # We place them at the beginning of the sequence dimension
                length = min(len(val_list), seq_len)
                if length > 0:
                    targets[idx, :length, t_i] = val_list[:length]

    # 5. Save to Cache
    save_dict = {"inputs": inputs, "partner_indices": partner_indices, "ids": ids}
    if targets is not None:
        save_dict["targets"] = targets

    np.savez_compressed(cache_path, **save_dict)
    print(f"Saved processed data to {cache_path}")

    return {
        "inputs": inputs,
        "partner_indices": partner_indices,
        "targets": targets,
        "ids": ids,
    }


# ==================================================================================
# DATASET CLASS
# ==================================================================================


class RNADataset(Dataset):
    def __init__(self, data_dict, mode="train"):
        self.inputs = torch.tensor(data_dict["inputs"], dtype=torch.float32)
        self.partner_indices = torch.tensor(
            data_dict["partner_indices"], dtype=torch.long
        )
        self.mode = mode
        self.ids = data_dict["ids"]

        if mode != "test" and data_dict["targets"] is not None:
            self.targets = torch.tensor(data_dict["targets"], dtype=torch.float32)
        else:
            self.targets = None

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        item = {
            "inputs": self.inputs[idx],  # (L, 18)
            "partner_indices": self.partner_indices[idx],  # (L,)
        }
        if self.targets is not None:
            item["targets"] = self.targets[idx]  # (L, 5)
        return item
