import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA degradation prediction.
    Serves sequence tokens, loop type tokens, signed pairing distances, and targets.
    """

    def __init__(self, sequences, loop_types, pair_dists, targets=None, ids=None):
        self.sequences = sequences
        self.loop_types = loop_types
        self.pair_dists = pair_dists
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Convert inputs to tensors
        seq = torch.tensor(self.sequences[idx], dtype=torch.long)
        loop = torch.tensor(self.loop_types[idx], dtype=torch.long)
        dist = torch.tensor(self.pair_dists[idx], dtype=torch.float32)

        # Create mask for scored positions (first 68)
        # The model processes 107 positions, but loss is only calculated on the first 68.
        mask = torch.zeros(Config.SEQ_LEN, dtype=torch.bool)
        mask[: Config.PRED_LEN] = True

        if self.targets is not None:
            # Retrieve targets (shape: 68, 3)
            target_data = self.targets[idx]

            # Pad targets to full sequence length (107, 3) for compatibility with model output shape
            # The loss function will use the mask to ignore the padded area (indices 68-106)
            padded_targets = torch.zeros(
                (Config.SEQ_LEN, Config.NUM_TARGETS), dtype=torch.float32
            )
            padded_targets[: Config.PRED_LEN, :] = torch.tensor(
                target_data, dtype=torch.float32
            )

            return seq, loop, dist, padded_targets, mask
        else:
            # For inference, return ID to map predictions back to sample
            sample_id = str(self.ids[idx])
            return seq, loop, dist, mask, sample_id


def get_structure_distance(structure):
    """
    Parses dot-bracket structure string to calculate signed pairing distances.
    For paired bases (i, j), the distance at i is (j - i) and at j is (i - j).
    Unpaired bases are assigned a distance of 0.
    """
    n = len(structure)
    dists = np.zeros(n, dtype=np.float32)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                # i is current (closing), j is popped (opening)
                # Distance is target_index - current_index
                dists[j] = i - j  # Positive (forward)
                dists[i] = j - i  # Negative (backward)
    return dists


def preprocess_data(df, mode="train"):
    """
    Converts raw DataFrame columns into numpy arrays suitable for the Dataset.
    """
    # 1. Process Sequences (Tokenization)
    # Map characters to integers based on Config.TOKEN_TO_ID
    seq_list = []
    for s in df["sequence"]:
        # Default to 0 if unknown char (should not happen in clean data)
        seq_ids = [Config.TOKEN_TO_ID.get(c, 0) for c in s]
        seq_list.append(seq_ids)
    sequences = np.array(seq_list, dtype=np.int32)

    # 2. Process Loop Types (Tokenization)
    loop_list = []
    for l in df["predicted_loop_type"]:
        loop_ids = [Config.LOOP_TO_ID.get(c, 0) for c in l]
        loop_list.append(loop_ids)
    loop_types = np.array(loop_list, dtype=np.int32)

    # 3. Process Structure (Signed Distances)
    dist_list = []
    for struct in df["structure"]:
        d = get_structure_distance(struct)
        dist_list.append(d)
    pair_dists = np.array(dist_list, dtype=np.float32)

    # 4. Process Targets (only for train/val)
    targets = None
    if mode in ["train", "val"]:
        target_arrays = []
        # Config.TARGET_COLS defines the specific columns to use (reactivity, deg_Mg_pH10, deg_Mg_50C)
        for col in Config.TARGET_COLS:
            # The columns in Parquet are stored as lists/arrays.
            # We stack them to create a matrix (N_samples, Seq_Scored)
            # We explicitly slice [:Config.PRED_LEN] to ensure consistency
            col_data = np.vstack(df[col].values)
            target_arrays.append(col_data[:, : Config.PRED_LEN])

        # Stack to shape (N, 68, 3)
        targets = np.stack(target_arrays, axis=2).astype(np.float32)

    # 5. IDs
    ids = df["id"].values.astype(str)

    return sequences, loop_types, pair_dists, targets, ids


def get_dataloaders(load_cached_data=True):
    """
    Generates DataLoaders for train, validation, and test sets.
    Implements caching using .npz files to speed up subsequent runs.
    """
    # Ensure working directory exists
    Config.setup()

    loaders = {}
    modes = ["train", "val", "test"]
    files = [Config.TRAIN_FILE, Config.VAL_FILE, Config.TEST_FILE]

    for mode, file_path in zip(modes, files):
        # Use distinct filenames for debug and full runs to avoid stale cache collisions
        suffix = "_debug" if Config.DEBUG else "_full"
        cache_path = os.path.join(Config.WORKING_DIR, f"{mode}_data{suffix}.npz")

        # Try loading from cache
        data_loaded = False
        if load_cached_data and os.path.exists(cache_path):
            try:
                print(f"Loading cached {mode} data from {cache_path}...")
                cached = np.load(cache_path)
                sequences = cached["sequences"]
                loop_types = cached["loop_types"]
                pair_dists = cached["pair_dists"]
                ids = cached["ids"]

                if mode in ["train", "val"]:
                    targets = cached["targets"]
                else:
                    targets = None
                data_loaded = True
            except Exception as e:
                print(f"Cache load failed for {mode}: {e}. Reprocessing...")

        # Process from scratch if cache missing or failed
        if not data_loaded:
            print(f"Processing {mode} data from {file_path}...")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Source file {file_path} not found.")

            df = pd.read_parquet(file_path)

            # Debugging: subset data if Config.DEBUG is True
            if Config.DEBUG:
                print(f"Debug mode: trimming {mode} data to 100 samples.")
                df = df.head(100)

            sequences, loop_types, pair_dists, targets, ids = preprocess_data(
                df, mode=mode
            )

            # Save to cache
            save_dict = {
                "sequences": sequences,
                "loop_types": loop_types,
                "pair_dists": pair_dists,
                "ids": ids,
            }
            if targets is not None:
                save_dict["targets"] = targets

            np.savez_compressed(cache_path, **save_dict)
            print(f"Saved {mode} data to {cache_path}")

        # Instantiate Dataset
        dataset = RNADataset(sequences, loop_types, pair_dists, targets, ids)

        # Instantiate DataLoader
        shuffle = mode == "train"
        loader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=shuffle,
            num_workers=Config.NUM_WORKERS,
            pin_memory=(Config.DEVICE == "cuda"),
            drop_last=(
                mode == "train"
            ),  # Drop last incomplete batch during training for stability
        )
        loaders[mode] = loader

    return loaders["train"], loaders["val"], loaders["test"]
