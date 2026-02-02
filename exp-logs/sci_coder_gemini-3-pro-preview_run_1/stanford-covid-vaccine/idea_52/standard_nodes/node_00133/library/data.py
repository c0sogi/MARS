import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything


def get_sinusoidal_encoding_np(positions, d_model):
    """
    Computes sinusoidal encodings for input positions.
    Handles both positive (absolute position) and negative (signed distance) integers.

    Args:
        positions: Numpy array of shape (N, L) containing integer positions/distances.
        d_model: Output embedding dimension.

    Returns:
        Numpy array of shape (N, L, d_model).
    """
    if positions.ndim == 1:
        positions = positions[np.newaxis, :]

    batch_size, seq_len = positions.shape
    pe = np.zeros((batch_size, seq_len, d_model), dtype=np.float32)

    # Calculate division term: 10000^(-2i/d_model)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    # Broadcast positions and div_term to compute arguments
    # positions: (N, L, 1)
    # div_term: (1, 1, d_model/2)
    args = positions[:, :, np.newaxis] * div_term[np.newaxis, np.newaxis, :]

    pe[:, :, 0::2] = np.sin(args)
    pe[:, :, 1::2] = np.cos(args)

    return pe


def parse_structure_distances(structure_strs, seq_len):
    """
    Parses dot-bracket structure strings to compute signed pairing distances.

    Args:
        structure_strs: List or array of structure strings.
        seq_len: Length of the sequences.

    Returns:
        Numpy array of shape (N, L) containing signed distances.
        Unpaired bases have distance 0.
        Paired (i, j) where i < j: index i has distance (j-i), index j has distance (i-j).
    """
    n_samples = len(structure_strs)
    distances = np.zeros((n_samples, seq_len), dtype=np.float32)

    for idx, struct in enumerate(structure_strs):
        stack = []
        for i, char in enumerate(struct):
            if char == "(":
                stack.append(i)
            elif char == ")":
                if stack:
                    j = stack.pop()
                    # Pair is (j, i) where j < i
                    dist = i - j
                    distances[idx, j] = dist
                    distances[idx, i] = -dist
    return distances


def encode_sequence(sequences, token_map, max_len):
    """
    Tokenizes sequences based on a provided mapping.
    """
    n_samples = len(sequences)
    ids = np.zeros((n_samples, max_len), dtype=np.int64)
    for i, seq in enumerate(sequences):
        # Truncate if necessary (though data should be fixed length)
        for j, char in enumerate(seq[:max_len]):
            ids[i, j] = token_map.get(char, 0)  # Default to 0 if unknown
    return ids


class RNADataset(Dataset):
    def __init__(self, data_dict, is_test=False):
        """
        Args:
            data_dict: Dictionary containing processed numpy arrays.
            is_test: Boolean flag indicating if this is the test set (no targets).
        """
        self.seq_ids = data_dict["seq_ids"]
        self.loop_ids = data_dict["loop_ids"]
        self.pair_emb = data_dict["pair_emb"]
        self.pos_emb = data_dict["pos_emb"]
        self.is_test = is_test

        if not self.is_test:
            self.targets = data_dict["targets"]
            self.masks = data_dict["masks"]

    def __len__(self):
        return len(self.seq_ids)

    def __getitem__(self, idx):
        item = {
            "seq_ids": torch.tensor(self.seq_ids[idx], dtype=torch.long),
            "loop_ids": torch.tensor(self.loop_ids[idx], dtype=torch.long),
            "pair_emb": torch.tensor(self.pair_emb[idx], dtype=torch.float32),
            "pos_emb": torch.tensor(self.pos_emb[idx], dtype=torch.float32),
        }

        if not self.is_test:
            item["targets"] = torch.tensor(self.targets[idx], dtype=torch.float32)
            item["mask"] = torch.tensor(self.masks[idx], dtype=torch.float32)

        return item


def process_and_cache_data(file_path, cache_path, load_cached_data=True, is_test=False):
    """
    Loads data from Parquet, processes it into tensors, and caches it as .npz.
    Strictly follows the caching logic: Load if exists and requested, else compute and save.
    """
    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            loaded = np.load(cache_path)
            data_dict = {
                "seq_ids": loaded["seq_ids"],
                "loop_ids": loaded["loop_ids"],
                "pair_emb": loaded["pair_emb"],
                "pos_emb": loaded["pos_emb"],
            }
            if not is_test:
                data_dict["targets"] = loaded["targets"]
                data_dict["masks"] = loaded["masks"]
            return data_dict
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Processing data from {file_path}...")
    df = pd.read_parquet(file_path)

    # A. Tokenization
    seq_ids = encode_sequence(
        df["sequence"].values, Config.TOKEN2ID_SEQ, Config.SEQ_LEN
    )
    loop_ids = encode_sequence(
        df["predicted_loop_type"].values, Config.TOKEN2ID_LOOP, Config.SEQ_LEN
    )

    # B. Structural Features (Pairing Distance)
    pair_dists = parse_structure_distances(df["structure"].values, Config.SEQ_LEN)
    pair_emb = get_sinusoidal_encoding_np(pair_dists, Config.EMB_PAIR_DIM)

    # C. Absolute Positional Encoding
    # Indices 0 to 106. Same for all samples.
    indices = np.arange(Config.SEQ_LEN)[np.newaxis, :]  # (1, 107)
    # Broadcast to (N, 107) for consistency in storage, though slightly redundant
    indices = np.repeat(indices, len(df), axis=0)
    pos_emb = get_sinusoidal_encoding_np(indices, Config.EMB_POS_DIM)

    data_dict = {
        "seq_ids": seq_ids,
        "loop_ids": loop_ids,
        "pair_emb": pair_emb,
        "pos_emb": pos_emb,
    }

    # D. Targets (Training/Validation only)
    if not is_test:
        # Initialize targets with zeros
        targets = np.zeros(
            (len(df), Config.SEQ_LEN, Config.NUM_CLASSES), dtype=np.float32
        )
        masks = np.zeros((len(df), Config.SEQ_LEN), dtype=np.float32)

        # Extract available targets
        # Note: Parquet metadata stores lists/arrays. We need to stack them.
        for i, col in enumerate(Config.TARGET_COLS):
            # df[col] contains lists/arrays of length seq_scored (68)
            # We stack them into (N, 68)
            col_data = np.vstack(df[col].values)
            # Assign to the first 68 positions
            targets[:, : Config.PRED_LEN, i] = col_data

        # Create mask: 1 for scored positions, 0 for others
        masks[:, : Config.PRED_LEN] = 1.0

        data_dict["targets"] = targets
        data_dict["masks"] = masks

    # 3. Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    print(f"Saving processed data to {cache_path}...")
    np.savez(cache_path, **data_dict)

    return data_dict


def get_dataloaders(
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
    debug=False,
):
    """
    Main entry point to get PyTorch DataLoaders.
    Handles caching and dataset creation.
    """
    seed_everything(Config.SEED)

    # Define Cache Paths
    train_cache = os.path.join(Config.WORKING_DIR, "train_data.npz")
    val_cache = os.path.join(Config.WORKING_DIR, "val_data.npz")
    test_cache = os.path.join(Config.WORKING_DIR, "test_data.npz")

    # 1. Process/Load Data
    train_data = process_and_cache_data(
        Config.TRAIN_FILE, train_cache, load_cached_data, is_test=False
    )
    val_data = process_and_cache_data(
        Config.VAL_FILE, val_cache, load_cached_data, is_test=False
    )
    test_data = process_and_cache_data(
        Config.TEST_FILE, test_cache, load_cached_data, is_test=True
    )

    # Debug Subsampling
    if debug:
        print("Debug mode: Subsampling datasets...")
        for d in [train_data, val_data]:
            for k in d.keys():
                d[k] = d[k][:100]
        for k in test_data.keys():
            test_data[k] = test_data[k][:50]

    # 2. Create Datasets
    train_dataset = RNADataset(train_data, is_test=False)
    val_dataset = RNADataset(val_data, is_test=False)
    test_dataset = RNADataset(test_data, is_test=True)

    # 3. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batch for stability
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
