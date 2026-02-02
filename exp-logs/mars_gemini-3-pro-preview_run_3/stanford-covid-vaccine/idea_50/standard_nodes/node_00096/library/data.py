import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


def get_structure_info(structure_str, seq_len):
    """
    Parses dot-bracket structure string to generate pair indices and masks.
    Returns:
        indices: np.array of shape (seq_len,). Paired bases point to their partner's index.
                 Unpaired bases point to 0 (handled by mask).
        mask: np.array of shape (seq_len,). 1.0 if paired, 0.0 if unpaired.
    """
    stack = []
    # Initialize indices to 0. Unpaired bases will point to 0.
    # The mask ensures these are zeroed out in the interaction module.
    indices = np.zeros(seq_len, dtype=np.int32)
    mask = np.zeros(seq_len, dtype=np.float32)

    for i, char in enumerate(structure_str):
        if i >= seq_len:
            break
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                start = stack.pop()
                indices[start] = i
                indices[i] = start
                mask[start] = 1.0
                mask[i] = 1.0
    return indices, mask


def one_hot_encode(seq, token_map, length):
    """
    One-hot encodes a sequence string based on a token map.
    """
    encoding = np.zeros((length, len(token_map)), dtype=np.float32)
    for i, char in enumerate(seq):
        if i >= length:
            break
        if char in token_map:
            encoding[i, token_map[char]] = 1.0
    return encoding


def process_data(df, cache_path, load_cached_data=True, is_test=False):
    """
    Processes the dataframe into numpy arrays, with caching.
    """
    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=True).item()
            # Verify length matches (in case subset size changed)
            if len(data["ids"]) == len(df):
                print(f"Loaded cached data from {cache_path}")
                return data
            else:
                print(
                    f"Cache size mismatch ({len(data['ids'])} vs {len(df)}). Recomputing..."
                )
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    print(f"Processing data for {len(df)} samples...")

    # 2. Pre-allocate arrays
    num_samples = len(df)
    seq_len = Config.SEQ_LEN
    pred_len = Config.PRED_LEN
    input_dim = Config.INPUT_DIM
    num_targets = Config.NUM_TARGETS

    # Features: Sequence (4) + Structure (3) + Loop (7) = 14
    features = np.zeros((num_samples, seq_len, input_dim), dtype=np.float32)
    pair_indices = np.zeros((num_samples, seq_len), dtype=np.int32)
    pair_masks = np.zeros((num_samples, seq_len), dtype=np.float32)
    targets = np.zeros((num_samples, pred_len, num_targets), dtype=np.float32)
    ids = []

    # 3. Iterate and Process
    for i, (_, row) in enumerate(df.iterrows()):
        sequence = row["sequence"]
        structure = row["structure"]
        loop_type = row["predicted_loop_type"]

        # --- Features ---
        # Sequence
        seq_enc = one_hot_encode(sequence, Config.TOKEN_MAP_SEQ, seq_len)
        # Structure
        struct_enc = one_hot_encode(structure, Config.TOKEN_MAP_STRUCT, seq_len)
        # Loop Type
        loop_enc = one_hot_encode(loop_type, Config.TOKEN_MAP_LOOP, seq_len)

        # Concatenate: (L, 4) + (L, 3) + (L, 7) -> (L, 14)
        features[i] = np.concatenate([seq_enc, struct_enc, loop_enc], axis=1)

        # --- Pair Indices & Masks ---
        p_idx, p_mask = get_structure_info(structure, seq_len)
        pair_indices[i] = p_idx
        pair_masks[i] = p_mask

        # --- Targets ---
        if not is_test:
            for t_idx, col in enumerate(Config.TARGET_COLS):
                val_list = row[col]
                # Ensure it's a list/array and slice/pad if necessary
                if isinstance(val_list, (list, np.ndarray)):
                    if len(val_list) > pred_len:
                        val_list = val_list[:pred_len]
                    targets[i, : len(val_list), t_idx] = val_list

        # --- ID ---
        ids.append(row["id"])

    # 4. Save to Cache
    data = {
        "features": features,
        "pair_indices": pair_indices,
        "pair_masks": pair_masks,
        "targets": targets,
        "ids": np.array(ids),
    }

    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, data)
    print(f"Saved processed data to {cache_path}")

    return data


class RNADataset(Dataset):
    def __init__(self, data):
        self.features = torch.from_numpy(data["features"])
        self.pair_indices = torch.from_numpy(data["pair_indices"]).long()
        self.pair_masks = torch.from_numpy(data["pair_masks"])
        self.targets = torch.from_numpy(data["targets"])
        self.ids = data["ids"]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return {
            "features": self.features[idx],
            "pair_indices": self.pair_indices[idx],
            "pair_masks": self.pair_masks[idx],
            "targets": self.targets[idx],
            "id": self.ids[idx],
        }


def get_dataloaders(
    load_cached_data=True, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
):
    """
    Loads data, creates datasets, and returns dataloaders.
    """
    # Load Parquet Files
    print("Loading metadata...")
    train_df = pd.read_parquet(Config.TRAIN_PATH)
    val_df = pd.read_parquet(Config.VAL_PATH)
    test_df = pd.read_parquet(Config.TEST_PATH)

    # Apply Subset if configured
    if Config.SUBSET_SIZE is not None:
        print(f"Subsetting data to {Config.SUBSET_SIZE} samples.")
        train_df = train_df.iloc[: Config.SUBSET_SIZE]
        val_df = val_df.iloc[: Config.SUBSET_SIZE]
        test_df = test_df.iloc[: Config.SUBSET_SIZE]

    # Process Data
    train_data = process_data(
        train_df, Config.TRAIN_CACHE_PATH, load_cached_data, is_test=False
    )
    val_data = process_data(
        val_df, Config.VAL_CACHE_PATH, load_cached_data, is_test=False
    )
    test_data = process_data(
        test_df, Config.TEST_CACHE_PATH, load_cached_data, is_test=True
    )

    # Create Datasets
    train_dataset = RNADataset(train_data)
    val_dataset = RNADataset(val_data)
    test_dataset = RNADataset(test_data)

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
