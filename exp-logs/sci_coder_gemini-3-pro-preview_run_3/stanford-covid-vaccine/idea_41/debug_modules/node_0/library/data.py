import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.utils import seed_everything

# Token mappings
NUCLEOTIDE_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCTURE_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_TYPE_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def get_couples(structure):
    """
    Parses a dot-bracket structure string to identify base pairs.
    Returns an array of shape (L,) where arr[i] is the index of the base paired with i.
    If i is unpaired, arr[i] = -1.
    """
    length = len(structure)
    pair_index = np.full(length, -1, dtype=np.int32)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                pair_index[i] = j
                pair_index[j] = i

    return pair_index


class RNADataset(Dataset):
    def __init__(self, features, pair_indices, targets=None, ids=None):
        """
        Args:
            features: (N, Seq_Len, Channels) - One-hot encoded features
            pair_indices: (N, Seq_Len) - Indices of paired bases (-1 if unpaired)
            targets: (N, Seq_Len, 5) - Target values (padded to Seq_Len)
            ids: (N,) - Sample IDs
        """
        self.features = features
        self.pair_indices = pair_indices
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Convert to torch tensors
        feat = torch.tensor(self.features[idx], dtype=torch.float32)
        pair_idx = torch.tensor(self.pair_indices[idx], dtype=torch.long)

        sample = {"features": feat, "pair_indices": pair_idx}

        if self.ids is not None:
            sample["id"] = self.ids[idx]

        if self.targets is not None:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            sample["targets"] = target

        return sample


def process_data(df):
    """
    Converts a pandas DataFrame into numpy arrays for the dataset.
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LEN

    # Initialize arrays
    # Channels: 4 (Seq) + 3 (Struct) + 7 (Loop) = 14
    features = np.zeros(
        (num_samples, seq_len, Config.NUM_INPUT_CHANNELS), dtype=np.float32
    )
    pair_indices = np.zeros((num_samples, seq_len), dtype=np.int32)

    # Check if targets exist
    has_targets = all(col in df.columns for col in Config.TARGET_COLS)
    targets = None
    if has_targets:
        targets = np.zeros((num_samples, seq_len, Config.NUM_TARGETS), dtype=np.float32)

    ids = df["id"].values

    # Process each sample
    # Note: Using a loop here is acceptable given the dataset size (~2k samples)
    # Vectorization is possible but string processing in numpy can be tricky.
    for i in range(num_samples):
        row = df.iloc[i]

        # 1. Sequence (One-Hot)
        seq = row["sequence"]
        for j, char in enumerate(seq):
            if char in NUCLEOTIDE_MAP:
                features[i, j, NUCLEOTIDE_MAP[char]] = 1.0

        # 2. Structure (One-Hot)
        struct = row["structure"]
        for j, char in enumerate(struct):
            if char in STRUCTURE_MAP:
                features[i, j, 4 + STRUCTURE_MAP[char]] = 1.0

        # 3. Loop Type (One-Hot)
        loop = row["predicted_loop_type"]
        for j, char in enumerate(loop):
            if char in LOOP_TYPE_MAP:
                features[i, j, 7 + LOOP_TYPE_MAP[char]] = 1.0

        # 4. Pair Indices
        pair_indices[i] = get_couples(struct)

        # 5. Targets
        if has_targets:
            # Targets are provided as lists/arrays of length seq_scored (68)
            # We place them into the (107,) array, leaving the rest as 0.
            for t_idx, col in enumerate(Config.TARGET_COLS):
                val = row[col]
                # Ensure val is a list or array
                if isinstance(val, (list, np.ndarray)):
                    length = len(val)
                    targets[i, :length, t_idx] = val

    return features, pair_indices, targets, ids


def load_data(split="train", load_cached_data=True, debug=False):
    """
    Loads data for a specific split. Handles caching to .npz files.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from cache.
        debug (bool): If True, subsets the data for debugging.

    Returns:
        RNADataset: The dataset object.
    """
    # Determine file paths
    cache_file = os.path.join(Config.WORKING_DIR, f"{split}_data.npz")

    if split == "train":
        source_path = Config.TRAIN_PARQUET
    elif split == "val":
        source_path = Config.VAL_PARQUET
    elif split == "test":
        source_path = Config.TEST_PARQUET
    else:
        raise ValueError(f"Unknown split: {split}")

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading {split} data from cache: {cache_file}")
        try:
            data = np.load(cache_file, allow_pickle=True)
            features = data["features"]
            pair_indices = data["pair_indices"]
            ids = data["ids"]
            # Targets might be None (stored as None object or missing)
            if (
                "targets" in data and not np.isnan(data["targets"]).all()
            ):  # Check if it was saved as valid
                targets = data["targets"]
            else:
                targets = None

            # If debug, slice the cached data
            if debug:
                features = features[: Config.DEBUG_SUBSET_SIZE]
                pair_indices = pair_indices[: Config.DEBUG_SUBSET_SIZE]
                ids = ids[: Config.DEBUG_SUBSET_SIZE]
                if targets is not None:
                    targets = targets[: Config.DEBUG_SUBSET_SIZE]

            return RNADataset(features, pair_indices, targets, ids)
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # Process from scratch
    print(f"Processing {split} data from metadata: {source_path}")
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Metadata file not found: {source_path}")

    df = pd.read_parquet(source_path)

    if debug:
        df = df.iloc[: Config.DEBUG_SUBSET_SIZE].reset_index(drop=True)

    features, pair_indices, targets, ids = process_data(df)

    # Save to cache (only if not debugging, to avoid overwriting full cache with subset)
    if not debug:
        print(f"Saving {split} data to cache: {cache_file}")
        # Use keyword arguments for savez
        save_dict = {"features": features, "pair_indices": pair_indices, "ids": ids}
        if targets is not None:
            save_dict["targets"] = targets
        else:
            # Save a placeholder or handle loading logic
            # np.savez doesn't handle None well, so we skip adding it to dict
            pass

        np.savez(cache_file, **save_dict)

    return RNADataset(features, pair_indices, targets, ids)


def collate_fn(batch):
    """
    Custom collate function to stack tensors.
    """
    features = torch.stack([item["features"] for item in batch])
    pair_indices = torch.stack([item["pair_indices"] for item in batch])

    out = {"features": features, "pair_indices": pair_indices}

    if "targets" in batch[0]:
        targets = torch.stack([item["targets"] for item in batch])
        out["targets"] = targets

    if "id" in batch[0]:
        out["id"] = [item["id"] for item in batch]

    return out
