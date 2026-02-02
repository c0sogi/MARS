import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# Mappings for One-Hot Encoding
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {".": 0, "(": 1, ")": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA degradation prediction.
    """

    def __init__(self, features, bpps_indices, bpps_mask, targets=None, ids=None):
        """
        Args:
            features (np.ndarray): Input features of shape (N, 107, 14).
            bpps_indices (np.ndarray): Paired indices of shape (N, 107).
            bpps_mask (np.ndarray): Mask for paired indices of shape (N, 107, 1).
            targets (np.ndarray, optional): Target values of shape (N, 68, 5).
            ids (np.ndarray, optional): Sample IDs.
        """
        self.features = features
        self.bpps_indices = bpps_indices
        self.bpps_mask = bpps_mask
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Convert numpy arrays to torch tensors
        item = {
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
            "bpps_indices": torch.tensor(self.bpps_indices[idx], dtype=torch.long),
            "bpps_mask": torch.tensor(self.bpps_mask[idx], dtype=torch.float32),
        }

        if self.targets is not None:
            item["targets"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        if self.ids is not None:
            item["ids"] = str(self.ids[idx])

        return item


def parse_structure(structure):
    """
    Parses a dot-bracket structure string to generate pair indices and a mask.

    Args:
        structure (str): Dot-bracket string (e.g., "..((..))..").

    Returns:
        tuple: (indices, mask)
            indices (np.ndarray): Array of shape (L,) where indices[i] = j if paired.
            mask (np.ndarray): Array of shape (L,) where mask[i] = 1.0 if paired, else 0.0.
    """
    length = len(structure)
    indices = np.zeros(length, dtype=np.int32)
    mask = np.zeros(length, dtype=np.float32)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                indices[i] = j
                indices[j] = i
                mask[i] = 1.0
                mask[j] = 1.0

    # Note: Unpaired positions have indices[i] = 0 and mask[i] = 0.0.
    # The interaction module should use mask * gathered_feat, so index 0 is safe.
    return indices, mask


def preprocess_data(sequences, structures, loop_types):
    """
    Converts raw sequences and structures into tensor-ready numpy arrays.
    """
    n_samples = len(sequences)
    seq_len = Config.SEQ_LEN

    # Initialize arrays
    # Channels: 4 (Seq) + 3 (Struct) + 7 (Loop) = 14
    features = np.zeros((n_samples, seq_len, 14), dtype=np.float32)
    bpps_indices = np.zeros((n_samples, seq_len), dtype=np.int32)
    bpps_mask = np.zeros((n_samples, seq_len, 1), dtype=np.float32)

    for i in range(n_samples):
        seq = sequences[i]
        struct = structures[i]
        loop = loop_types[i]

        # 1. Sequence One-Hot
        for j, char in enumerate(seq):
            if char in SEQ_MAP:
                features[i, j, SEQ_MAP[char]] = 1.0

        # 2. Structure One-Hot
        for j, char in enumerate(struct):
            if char in STRUCT_MAP:
                features[i, j, 4 + STRUCT_MAP[char]] = 1.0

        # 3. Loop Type One-Hot
        for j, char in enumerate(loop):
            if char in LOOP_MAP:
                features[i, j, 7 + LOOP_MAP[char]] = 1.0

        # 4. Adjacency Map & Mask
        inds, msk = parse_structure(struct)
        bpps_indices[i] = inds
        bpps_mask[i, :, 0] = msk

    return features, bpps_indices, bpps_mask


def get_loader(
    split,
    batch_size=Config.BATCH_SIZE,
    shuffle=True,
    load_cached_data=True,
    num_workers=Config.NUM_WORKERS,
):
    """
    Retrieves a DataLoader for the specified split, handling caching and preprocessing.

    Args:
        split (str): 'train', 'val', or 'test'.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the data.
        load_cached_data (bool): Whether to attempt loading from cache.
        num_workers (int): Number of worker threads for DataLoader.

    Returns:
        DataLoader: The configured PyTorch DataLoader.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(Config.CACHE_DIR, f"{split}_data.npz")

    data_loaded = False
    features, bpps_indices, bpps_mask, targets, ids = None, None, None, None, None

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading {split} data from cache: {cache_path}")
            data = np.load(cache_path, allow_pickle=True)
            features = data["features"]
            bpps_indices = data["bpps_indices"]
            bpps_mask = data["bpps_mask"]
            ids = data["ids"]

            if "targets" in data:
                targets = data["targets"]

            data_loaded = True
        except Exception as e:
            print(f"Failed to load cache for {split}: {e}. Recomputing...")
            data_loaded = False

    # 2. Process from scratch if needed
    if not data_loaded:
        print(f"Processing {split} data from scratch...")

        # Determine file path
        if split == "train":
            file_path = Config.TRAIN_DATA_PATH
        elif split == "val":
            file_path = Config.VAL_DATA_PATH
        elif split == "test":
            file_path = Config.TEST_DATA_PATH
        else:
            raise ValueError(f"Invalid split: {split}")

        # Load Parquet
        df = pd.read_parquet(file_path)

        # Handle Debug Mode
        if Config.DEBUG:
            print(f"DEBUG MODE: Using subset of {Config.DEBUG_SUBSET_SIZE} samples.")
            df = df.iloc[: Config.DEBUG_SUBSET_SIZE].copy()

        # Extract Raw Data
        sequences = df["sequence"].tolist()
        structures = df["structure"].tolist()
        loop_types = df["predicted_loop_type"].tolist()
        ids = df["id"].values

        # Preprocess Features
        features, bpps_indices, bpps_mask = preprocess_data(
            sequences, structures, loop_types
        )

        # Process Targets (only for train/val)
        if split in ["train", "val"]:
            # Targets are stored as lists in the dataframe columns.
            # We extract them and stack them into (N, 68, 5).
            target_list = []
            for col in Config.TARGET_COLS:
                # df[col] is a Series of lists. tolist() converts to list of lists.
                col_data = np.array(df[col].tolist(), dtype=np.float32)
                target_list.append(col_data)

            targets = np.stack(target_list, axis=2)  # (N, 68, 5)

        # Save to Cache
        save_dict = {
            "features": features,
            "bpps_indices": bpps_indices,
            "bpps_mask": bpps_mask,
            "ids": ids,
        }
        if targets is not None:
            save_dict["targets"] = targets

        np.savez_compressed(cache_path, **save_dict)
        print(f"Saved {split} data to cache: {cache_path}")

    # 3. Create Dataset and DataLoader
    dataset = RNADataset(features, bpps_indices, bpps_mask, targets, ids)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=(
            split == "train"
        ),  # Drop last batch only during training to maintain stability
    )

    return loader
