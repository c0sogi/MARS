import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything

# Token maps
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA degradation prediction.
    """

    def __init__(
        self, sequences, loop_types, pair_dists, targets=None, masks=None, ids=None
    ):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.loop_types = torch.tensor(loop_types, dtype=torch.long)
        self.pair_dists = torch.tensor(
            pair_dists, dtype=torch.float32
        )  # Float for sinusoidal encoding input

        # Targets and masks are present for train/val, potentially None for test
        if targets is not None:
            self.targets = torch.tensor(targets, dtype=torch.float32)
        else:
            self.targets = None

        if masks is not None:
            self.masks = torch.tensor(masks, dtype=torch.float32)
        else:
            self.masks = None

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

        if self.masks is not None:
            sample["mask"] = self.masks[idx]

        if self.ids is not None:
            sample["id"] = self.ids[idx]

        return sample


def parse_structure_to_distance(structure, seq_len):
    """
    Parses dot-bracket structure to signed pairing distance array.
    If (i, j) are paired, dist[i] = j - i, dist[j] = i - j.
    Unpaired bases have dist 0.
    """
    n = len(structure)
    # Ensure we handle length mismatches if any (though data should be clean)
    limit = min(n, seq_len)

    dists = np.zeros(seq_len, dtype=np.float32)
    stack = []

    for i in range(limit):
        char = structure[i]
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                start_idx = stack.pop()
                # Signed distance
                dists[start_idx] = i - start_idx
                dists[i] = start_idx - i

    return dists


def process_dataframe(df, mode="train"):
    """
    Extracts features and targets from the dataframe.
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LEN

    # Pre-allocate arrays
    sequences = np.zeros((num_samples, seq_len), dtype=np.int32)
    loop_types = np.zeros((num_samples, seq_len), dtype=np.int32)
    pair_dists = np.zeros((num_samples, seq_len), dtype=np.float32)

    # Targets and Masks
    # Train/Val have targets. Test does not.
    # We output targets with shape (N, 107, 3).
    # Raw data has 68 values. We pad with 0.
    targets = None
    masks = None

    if mode != "test":
        targets = np.zeros((num_samples, seq_len, 3), dtype=np.float32)
        masks = np.zeros((num_samples, seq_len), dtype=np.float32)

    ids = df["id"].values

    # Iterate and fill
    # Using iterrows is slow, but robust for list columns.
    # Given dataset size (~2k), it's acceptable.
    for idx, row in df.iterrows():
        # 1. Sequence
        seq_str = row["sequence"]
        sequences[idx, :] = [SEQ_MAP.get(c, 0) for c in seq_str[:seq_len]]

        # 2. Loop Type
        loop_str = row["predicted_loop_type"]
        loop_types[idx, :] = [LOOP_MAP.get(c, 0) for c in loop_str[:seq_len]]

        # 3. Structure (Pairing Distance)
        struct_str = row["structure"]
        pair_dists[idx, :] = parse_structure_to_distance(struct_str, seq_len)

        # 4. Targets (if not test)
        if mode != "test":
            # Extract the 3 specific columns
            # These are lists in the parquet dataframe
            t_react = row["reactivity"]
            t_mg_ph10 = row["deg_Mg_pH10"]
            t_mg_50c = row["deg_Mg_50C"]

            # Determine valid length (usually 68)
            # Some rows might have different lengths if data is dirty, but metadata checks passed.
            # We trust seq_scored or len(t_react).
            valid_len = len(t_react)

            # Fill targets
            targets[idx, :valid_len, 0] = t_react
            targets[idx, :valid_len, 1] = t_mg_ph10
            targets[idx, :valid_len, 2] = t_mg_50c

            # Fill mask
            masks[idx, :valid_len] = 1.0

    return sequences, loop_types, pair_dists, targets, masks, ids


def load_or_process_data(file_path, cache_path, mode="train", load_cached_data=True):
    """
    Loads data from cache or processes from source parquet file.
    """
    # Ensure cache directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {mode} data from {cache_path}...")
        data = np.load(cache_path, allow_pickle=True)
        sequences = data["sequences"]
        loop_types = data["loop_types"]
        pair_dists = data["pair_dists"]
        ids = data["ids"]

        if mode != "test":
            targets = data["targets"]
            masks = data["masks"]
            return sequences, loop_types, pair_dists, targets, masks, ids
        else:
            return sequences, loop_types, pair_dists, None, None, ids

    # Process from scratch
    print(f"Processing {mode} data from {file_path}...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Source file {file_path} not found.")

    df = pd.read_parquet(file_path)

    # Debug mode: subset data
    if Config.DEBUG:
        print(f"DEBUG MODE: Slicing {mode} data to {Config.DEBUG_SAMPLES} samples.")
        df = df.head(Config.DEBUG_SAMPLES)

    sequences, loop_types, pair_dists, targets, masks, ids = process_dataframe(
        df, mode=mode
    )

    # Save to cache
    print(f"Saving {mode} data to {cache_path}...")
    save_dict = {
        "sequences": sequences,
        "loop_types": loop_types,
        "pair_dists": pair_dists,
        "ids": ids,
    }
    if targets is not None:
        save_dict["targets"] = targets
        save_dict["masks"] = masks

    np.savez_compressed(cache_path, **save_dict)

    return sequences, loop_types, pair_dists, targets, masks, ids


def get_dataloaders(load_cached_data=True):
    """
    Main function to get train, val, and test dataloaders.
    """
    seed_everything(Config.SEED)

    # Define cache paths
    cache_train = os.path.join(Config.WORKING_DIR, "train_data.npz")
    cache_val = os.path.join(Config.WORKING_DIR, "val_data.npz")
    cache_test = os.path.join(Config.WORKING_DIR, "test_data.npz")

    # 1. Train Data
    seq_tr, loop_tr, pair_tr, y_tr, mask_tr, ids_tr = load_or_process_data(
        Config.TRAIN_FILE, cache_train, mode="train", load_cached_data=load_cached_data
    )
    train_dataset = RNADataset(seq_tr, loop_tr, pair_tr, y_tr, mask_tr, ids_tr)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    # 2. Val Data
    seq_val, loop_val, pair_val, y_val, mask_val, ids_val = load_or_process_data(
        Config.VAL_FILE, cache_val, mode="val", load_cached_data=load_cached_data
    )
    val_dataset = RNADataset(seq_val, loop_val, pair_val, y_val, mask_val, ids_val)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    # 3. Test Data
    seq_test, loop_test, pair_test, _, _, ids_test = load_or_process_data(
        Config.TEST_FILE, cache_test, mode="test", load_cached_data=load_cached_data
    )
    test_dataset = RNADataset(
        seq_test, loop_test, pair_test, targets=None, masks=None, ids=ids_test
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    return train_loader, val_loader, test_loader
