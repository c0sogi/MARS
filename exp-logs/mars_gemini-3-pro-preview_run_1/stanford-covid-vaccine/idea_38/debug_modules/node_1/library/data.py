import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# Token maps for integer encoding
NUCLEOTIDE_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
LOOP_TYPE_MAP = {"B": 0, "E": 1, "H": 2, "I": 3, "M": 4, "S": 5, "X": 6}


def parse_structure_to_pairs(structure):
    """
    Parses a dot-bracket structure string to find base pairs.
    Returns a dictionary mapping index i to index j for all pairs (i, j).
    """
    pairs = {}
    stack = []
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                pairs[j] = i
                pairs[i] = j
    return pairs


def get_pair_distance_vector(structure, seq_len):
    """
    Generates a vector of length seq_len.
    For each position i:
      - If paired with j, value is (j - i).
      - If unpaired, value is 0.
    """
    pairs = parse_structure_to_pairs(structure)
    # Initialize with 0
    dist_vector = np.zeros(seq_len, dtype=np.float32)

    for i in range(seq_len):
        if i in pairs:
            j = pairs[i]
            dist_vector[i] = float(j - i)

    return dist_vector


def token_encode(seq, token_map):
    """Encodes a string sequence into a list of integers based on a map."""
    return [token_map.get(c, 0) for c in seq]


class RNADataset(Dataset):
    def __init__(self, sequences, loop_types, pair_dists, targets=None, ids=None):
        """
        Args:
            sequences: (N, L) tensor of integer encoded sequences
            loop_types: (N, L) tensor of integer encoded loop types
            pair_dists: (N, L) tensor of float pair distances
            targets: (N, L, 3) tensor of targets (optional)
            ids: list of sequence IDs
        """
        self.sequences = sequences
        self.loop_types = loop_types
        self.pair_dists = pair_dists
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sample = {
            "sequence": self.sequences[idx],
            "loop_type": self.loop_types[idx],
            "pair_dist": self.pair_dists[idx],
        }

        if self.targets is not None:
            sample["target"] = self.targets[idx]

        if self.ids is not None:
            sample["id"] = self.ids[idx]

        return sample


def process_dataframe(df, mode="train"):
    """
    Extracts features and targets from the dataframe.
    Returns tensors ready for the dataset.
    """
    sequences = []
    loop_types = []
    pair_dists = []
    targets = []
    ids = df["id"].tolist()

    for _, row in df.iterrows():
        # 1. Sequence Encoding
        seq_ints = token_encode(row["sequence"], NUCLEOTIDE_MAP)
        sequences.append(seq_ints)

        # 2. Loop Type Encoding
        loop_ints = token_encode(row["predicted_loop_type"], LOOP_TYPE_MAP)
        loop_types.append(loop_ints)

        # 3. Pair Distance
        struct = row["structure"]
        seq_len = len(row["sequence"])
        p_dist = get_pair_distance_vector(struct, seq_len)
        pair_dists.append(p_dist)

        # 4. Targets (only for train/val)
        if mode in ["train", "val"]:
            # Extract target lists
            t_reactivity = np.array(row["reactivity"], dtype=np.float32)
            t_mg_ph10 = np.array(row["deg_Mg_pH10"], dtype=np.float32)
            t_mg_50c = np.array(row["deg_Mg_50C"], dtype=np.float32)

            # Stack targets: (68, 3)
            target_matrix = np.stack([t_reactivity, t_mg_ph10, t_mg_50c], axis=1)

            # Pad to sequence length (107) with zeros
            # The loss function will mask these out based on seq_scored
            curr_len = target_matrix.shape[0]
            pad_len = seq_len - curr_len
            if pad_len > 0:
                padding = np.zeros((pad_len, 3), dtype=np.float32)
                target_matrix = np.concatenate([target_matrix, padding], axis=0)

            targets.append(target_matrix)

    # Convert lists to PyTorch tensors
    sequences_tensor = torch.tensor(sequences, dtype=torch.long)
    loop_types_tensor = torch.tensor(loop_types, dtype=torch.long)
    pair_dists_tensor = torch.tensor(np.array(pair_dists), dtype=torch.float32)

    if mode in ["train", "val"]:
        targets_tensor = torch.tensor(np.array(targets), dtype=torch.float32)
        return (
            sequences_tensor,
            loop_types_tensor,
            pair_dists_tensor,
            targets_tensor,
            ids,
        )
    else:
        return sequences_tensor, loop_types_tensor, pair_dists_tensor, None, ids


def get_dataloaders(load_cached_data=Config.LOAD_CACHED_DATA):
    """
    Main entry point to get DataLoaders.
    Handles caching logic to avoid re-processing data on every run.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    modes = ["train", "val", "test"]
    datasets = {}

    for mode in modes:
        cache_path = os.path.join(Config.CACHE_DIR, f"{mode}_data.pt")

        data_loaded = False
        # Attempt to load from cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                print(f"Loading cached {mode} data from {cache_path}...")
                cached_data = torch.load(cache_path)
                sequences = cached_data["sequences"]
                loop_types = cached_data["loop_types"]
                pair_dists = cached_data["pair_dists"]
                targets = cached_data["targets"]
                ids = cached_data["ids"]
                data_loaded = True
            except Exception as e:
                print(f"Failed to load cache for {mode}: {e}. Reprocessing...")

        # Process from scratch if not loaded
        if not data_loaded:
            print(f"Processing {mode} data from metadata...")
            # Select correct metadata file
            if mode == "train":
                meta_path = Config.TRAIN_METADATA
            elif mode == "val":
                meta_path = Config.VAL_METADATA
            else:
                meta_path = Config.TEST_METADATA

            if not os.path.exists(meta_path):
                raise FileNotFoundError(f"Metadata file not found: {meta_path}")

            df = pd.read_parquet(meta_path)

            sequences, loop_types, pair_dists, targets, ids = process_dataframe(
                df, mode=mode
            )

            # Save to cache
            print(f"Saving {mode} data to cache...")
            torch.save(
                {
                    "sequences": sequences,
                    "loop_types": loop_types,
                    "pair_dists": pair_dists,
                    "targets": targets,
                    "ids": ids,
                },
                cache_path,
            )

        # Initialize Dataset
        datasets[mode] = RNADataset(sequences, loop_types, pair_dists, targets, ids)

    # Create DataLoaders
    # Pin memory enables faster data transfer to CUDA
    pin_memory = Config.DEVICE == "cuda"

    train_loader = DataLoader(
        datasets["train"],
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        datasets["val"],
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        datasets["test"],
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
