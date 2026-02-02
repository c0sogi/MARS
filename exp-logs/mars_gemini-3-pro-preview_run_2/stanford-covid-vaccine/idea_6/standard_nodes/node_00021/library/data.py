import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.utils import parse_structure_pairs, parse_list_column


def process_data(csv_path, mode="train", load_cached_data=True):
    """
    Processes raw CSV data into numpy tensors with Stacking-Aware features.

    Features (Total Dim: 29):
    1. Sequence One-Hot (4): A, G, C, U
    2. Structure One-Hot (3): (, ), .
    3. Loop Type One-Hot (7): S, M, I, B, H, E, X
    4. Partner Triplet (15): For paired base i->j, one-hot of seq[j-1], seq[j], seq[j+1].
                             3 positions * 5 classes (A, G, C, U, Pad).

    Args:
        csv_path (str): Path to the source CSV file.
        mode (str): 'train', 'val', or 'test'. Used for naming cache files.
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        tuple: (inputs, targets, ids)
            inputs: np.ndarray [N, 107, 29]
            targets: np.ndarray [N, 107, 5]
            ids: np.ndarray [N]
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(Config.CACHE_DIR, f"{mode}_processed.npz")

    # 1. Load from cache if available
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached {mode} data from {cache_file}...")
        data = np.load(cache_file)
        return data["inputs"], data["targets"], data["ids"]

    # 2. Process from scratch
    print(f"Processing {mode} data from {csv_path}...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Mappings
    seq_map = {"A": 0, "G": 1, "C": 2, "U": 3}
    struct_map = {"(": 0, ")": 1, ".": 2}
    loop_map = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

    num_samples = len(df)

    # Initialize tensors
    # Dimensions: 4 (Seq) + 3 (Struct) + 7 (Loop) + 15 (Partner) = 29
    inputs = np.zeros((num_samples, Config.SEQ_LEN, Config.INPUT_DIM), dtype=np.float32)
    targets = np.zeros(
        (num_samples, Config.SEQ_LEN, Config.OUTPUT_DIM), dtype=np.float32
    )
    ids = df["id"].values

    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Feature Engineering Loop
    for idx, row in df.iterrows():
        seq = row["sequence"]
        struct = row["structure"]
        loop = row["predicted_loop_type"]

        # Get pairing information: {index: paired_index}
        pairs = parse_structure_pairs(struct)

        # Iterate over sequence length
        for i in range(Config.SEQ_LEN):
            # --- 1. Basic Features ---

            # Sequence (0-3)
            if i < len(seq):
                s_char = seq[i]
                if s_char in seq_map:
                    inputs[idx, i, seq_map[s_char]] = 1.0

            # Structure (4-6)
            if i < len(struct):
                st_char = struct[i]
                if st_char in struct_map:
                    inputs[idx, i, 4 + struct_map[st_char]] = 1.0

            # Loop Type (7-13)
            if i < len(loop):
                l_char = loop[i]
                if l_char in loop_map:
                    inputs[idx, i, 7 + loop_map[l_char]] = 1.0

            # --- 2. Partner Triplet Features (14-28) ---
            # Offset starts at 4 + 3 + 7 = 14
            base_offset = 14

            if i in pairs:
                j = pairs[i]
                # We want context around the partner j: [j-1, j, j+1]
                triplet_indices = [j - 1, j, j + 1]

                for k, t_idx in enumerate(triplet_indices):
                    # Determine feature index for this position in the triplet
                    # 0=A, 1=G, 2=C, 3=U, 4=Pad
                    feat_idx = 4  # Default to Pad

                    if 0 <= t_idx < len(seq):
                        t_char = seq[t_idx]
                        if t_char in seq_map:
                            feat_idx = seq_map[t_char]

                    # Calculate exact channel index
                    # Each of the 3 positions has 5 possible classes
                    # channel = base_offset + (position_in_triplet * 5) + class_index
                    channel = base_offset + (k * 5) + feat_idx
                    inputs[idx, i, channel] = 1.0

        # --- 3. Targets ---
        # Only process targets if they exist (train/val)
        # Test set might not have them, or we might just fill zeros
        if mode in ["train", "val"]:
            for t_i, col in enumerate(target_cols):
                if col in row:
                    val_arr = parse_list_column(row[col])
                    length = len(val_arr)
                    # Copy available data; rest remains 0 (padded)
                    if length > 0:
                        targets[idx, :length, t_i] = val_arr

    # 3. Save to cache
    np.savez(cache_file, inputs=inputs, targets=targets, ids=ids)
    print(f"Saved processed {mode} data to {cache_file}")

    return inputs, targets, ids


class RNADataset(Dataset):
    def __init__(self, inputs, targets, ids, mode="train"):
        """
        PyTorch Dataset for RNA data.

        Args:
            inputs (np.ndarray): Input features [N, Seq, Dim]
            targets (np.ndarray): Target values [N, Seq, 5]
            ids (np.ndarray): Sequence IDs [N]
            mode (str): Mode of the dataset
        """
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.ids = ids
        self.mode = mode

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], self.ids[idx]
