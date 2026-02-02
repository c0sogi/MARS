import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config


def get_structure_map(structure):
    """
    Parses dot-bracket structure to get paired indices.
    Returns an array where index i contains the index of its pair j, or -1 if unpaired.
    """
    mapping = np.full(len(structure), -1, dtype=int)
    stack = []
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                mapping[i] = j
                mapping[j] = i
    return mapping


def process_data(df, is_test=False):
    """
    Processes the dataframe into numpy arrays for the model.
    Generates One-Hot encodings, Partner Identity, and Anchored Targets.
    """
    # 1. Sequences (One-Hot): A, G, C, U -> 4 channels
    seq_map = {"A": 0, "G": 1, "C": 2, "U": 3}
    sequences = []
    for seq in df["sequence"]:
        vec = np.zeros((Config.SEQ_LENGTH, 4), dtype=np.float32)
        for i, char in enumerate(seq):
            if char in seq_map:
                vec[i, seq_map[char]] = 1.0
        sequences.append(vec)
    sequences = np.array(sequences)

    # 2. Structures (One-Hot): ., (, ) -> 3 channels
    # Also generate Pair Maps for structural attention
    struct_map = {".": 0, "(": 1, ")": 2}
    structures = []
    pair_maps = []
    for struct in df["structure"]:
        vec = np.zeros((Config.SEQ_LENGTH, 3), dtype=np.float32)
        for i, char in enumerate(struct):
            if char in struct_map:
                vec[i, struct_map[char]] = 1.0
        structures.append(vec)
        pair_maps.append(get_structure_map(struct))
    structures = np.array(structures)
    pair_maps = np.array(pair_maps)

    # 3. Loop Types (One-Hot): S, M, I, B, H, E, X -> 7 channels
    loop_map = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}
    loops = []
    for lp in df["predicted_loop_type"]:
        vec = np.zeros((Config.SEQ_LENGTH, 7), dtype=np.float32)
        for i, char in enumerate(lp):
            if char in loop_map:
                vec[i, loop_map[char]] = 1.0
        loops.append(vec)
    loops = np.array(loops)

    # 4. Partner Identity: Explicit injection of the paired base's identity -> 4 channels
    partner_identities = []
    for i in range(len(sequences)):
        p_id = np.zeros((Config.SEQ_LENGTH, 4), dtype=np.float32)
        pm = pair_maps[i]
        seq = sequences[i]

        # Identify valid pairs
        valid = pm != -1
        if np.any(valid):
            # Gather the sequence vector from the paired position
            p_id[valid] = seq[pm[valid]]

        partner_identities.append(p_id)
    partner_identities = np.array(partner_identities)

    # 5. Targets (Anchored)
    targets = None
    if not is_test:
        # Initialize with zeros (Neutral Baseline for Anchoring)
        targets = np.zeros((len(df), Config.SEQ_LENGTH, 5), dtype=np.float32)

        for idx, col in enumerate(Config.TARGET_COLS):
            # Parse stringified lists safely
            vals = df[col].apply(
                lambda x: (
                    np.array(ast.literal_eval(x)) if isinstance(x, str) else np.array(x)
                )
            )
            for i, val in enumerate(vals):
                # Fill the available ground truth (usually first 68 positions)
                length = min(len(val), Config.SEQ_LENGTH)
                targets[i, :length, idx] = val[:length]
                # The tail (68-107) remains 0.0, implementing Boundary Anchoring

    # Return dictionary containing all processed arrays
    return {
        "sequence": sequences,
        "structure": structures,
        "loop": loops,
        "partner_identity": partner_identities,
        "pair_map": pair_maps,
        "targets": targets,
        "ids": df["id"].values,
    }


def get_dataset(mode="train", load_cached_data=True):
    """
    Loads data from cache or computes it from scratch.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: A dictionary containing processed numpy arrays.
    """
    # Ensure working directory exists
    os.makedirs(Config.IDEA_DIR, exist_ok=True)

    # Define cache filename based on the idea version
    cache_file = os.path.join(Config.IDEA_DIR, f"{mode}_data_ahc_hdn_v1.npz")

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached {mode} data from {cache_file}...")
        try:
            data = np.load(cache_file, allow_pickle=True)
            return dict(data)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Process from Scratch
    print(f"Processing {mode} data from scratch...")
    if mode == "test":
        df = pd.read_csv(os.path.join(Config.METADATA_DIR, "test.csv"))
        data = process_data(df, is_test=True)
    else:
        df = pd.read_csv(os.path.join(Config.METADATA_DIR, f"{mode}.csv"))
        data = process_data(df, is_test=False)

    # 3. Save Cache
    print(f"Saving {mode} data to cache...")
    np.savez(cache_file, **data)

    return data


class RNADataset(Dataset):
    def __init__(self, data, mode="train"):
        self.sequence = torch.FloatTensor(data["sequence"])
        self.structure = torch.FloatTensor(data["structure"])
        self.loop = torch.FloatTensor(data["loop"])
        self.partner_identity = torch.FloatTensor(data["partner_identity"])
        self.pair_map = torch.LongTensor(data["pair_map"])
        self.mode = mode

        if mode != "test":
            self.targets = torch.FloatTensor(data["targets"])

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, idx):
        # Concatenate static features:
        # Seq(4) + Struct(3) + Loop(7) + Partner(4) = 18 channels
        features = torch.cat(
            [
                self.sequence[idx],
                self.structure[idx],
                self.loop[idx],
                self.partner_identity[idx],
            ],
            dim=1,
        )

        pair_map = self.pair_map[idx]

        if self.mode == "test":
            return features, pair_map
        else:
            return features, pair_map, self.targets[idx]
