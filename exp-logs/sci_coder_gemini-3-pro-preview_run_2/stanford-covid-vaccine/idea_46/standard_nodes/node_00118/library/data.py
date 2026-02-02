import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config


def get_structure_adj(structure, seq_len):
    """
    Parses dot-bracket structure to get partner indices.
    Returns an array of length seq_len where arr[i] is the index of the base paired with i,
    or -1 if unpaired.
    """
    adj = np.full(seq_len, -1, dtype=np.int32)
    stack = []
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                adj[i] = j
                adj[j] = i
    return adj


def parse_list_col(x):
    """
    Parses a string representation of a list into a numpy array.
    Returns an array of zeros if parsing fails.
    """
    try:
        return np.array(ast.literal_eval(x), dtype=np.float32)
    except:
        return np.zeros(Config.PRED_LEN, dtype=np.float32)


def process_data(mode="train", load_cached_data=True):
    """
    Loads metadata, processes sequences/structures, and caches the result.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        dict: A dictionary containing numpy arrays for ids, seq, struct, loop,
              partner_idx, partner_id, and targets (if not test).
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Define cache filename specific to this idea version
    cache_file = os.path.join(Config.CACHE_DIR, f"{mode}_data_sdf_rn_v1.npz")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached {mode} data from {cache_file}...")
        try:
            return dict(np.load(cache_file, allow_pickle=True))
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from scratch
    print(f"Processing {mode} data from scratch...")
    csv_path = os.path.join(Config.METADATA_DIR, f"{mode}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Mappings
    seq_map = {c: i for i, c in enumerate("AGUC")}
    struct_map = {c: i for i, c in enumerate(".()")}
    loop_map = {c: i for i, c in enumerate("SMIBHEX")}

    # Containers
    ids = df["id"].values
    seq_encoded = []
    struct_encoded = []
    loop_encoded = []
    partner_indices = []
    partner_identities = []  # Identity of the paired base

    targets = []

    for idx, row in df.iterrows():
        # Sequence
        seq_ints = [seq_map.get(c, 0) for c in row["sequence"]]
        seq_encoded.append(seq_ints)

        # Structure
        struct_ints = [struct_map.get(c, 0) for c in row["structure"]]
        struct_encoded.append(struct_ints)

        # Loop
        loop_ints = [loop_map.get(c, 0) for c in row["predicted_loop_type"]]
        loop_encoded.append(loop_ints)

        # Adjacency / Partner
        # Note: We use Config.SEQ_LEN which is 107
        adj = get_structure_adj(row["structure"], Config.SEQ_LEN)
        partner_indices.append(adj)

        # Partner Identity
        # If i is paired with j, identity is seq[j].
        # If unpaired, use a special token (4) for "No Partner".
        p_ids = []
        for i, neighbor in enumerate(adj):
            if neighbor != -1:
                p_ids.append(seq_ints[neighbor])
            else:
                p_ids.append(4)  # 4 = None/Unpaired
        partner_identities.append(p_ids)

        # Targets (only for train/val)
        if mode != "test":
            t_list = []
            for col in Config.ALL_TARGETS:
                val = parse_list_col(row[col])
                # Pad to SEQ_LEN (model outputs 107, but we only score first 68)
                padded = np.zeros(Config.SEQ_LEN, dtype=np.float32)
                # Handle cases where val might be empty or shorter than expected
                length = min(len(val), Config.SEQ_LEN)
                padded[:length] = val[:length]
                t_list.append(padded)
            targets.append(np.stack(t_list, axis=1))  # (L, 5)

    # Convert to numpy
    data = {
        "ids": ids,
        "seq": np.array(seq_encoded, dtype=np.int32),
        "struct": np.array(struct_encoded, dtype=np.int32),
        "loop": np.array(loop_encoded, dtype=np.int32),
        "partner_idx": np.array(partner_indices, dtype=np.int32),
        "partner_id": np.array(partner_identities, dtype=np.int32),
    }

    if mode != "test":
        data["targets"] = np.array(targets, dtype=np.float32)

    # Save to cache
    np.savez_compressed(cache_file, **data)
    print(f"Saved processed data to {cache_file}")

    return data


class RNADataset(Dataset):
    def __init__(self, data, mode="train"):
        self.data = data
        self.mode = mode

    def __len__(self):
        return len(self.data["ids"])

    def __getitem__(self, idx):
        item = {
            "seq": torch.tensor(self.data["seq"][idx], dtype=torch.long),
            "struct": torch.tensor(self.data["struct"][idx], dtype=torch.long),
            "loop": torch.tensor(self.data["loop"][idx], dtype=torch.long),
            "partner_idx": torch.tensor(
                self.data["partner_idx"][idx], dtype=torch.long
            ),
            "partner_id": torch.tensor(self.data["partner_id"][idx], dtype=torch.long),
        }

        if self.mode != "test":
            item["targets"] = torch.tensor(
                self.data["targets"][idx], dtype=torch.float32
            )

        return item
