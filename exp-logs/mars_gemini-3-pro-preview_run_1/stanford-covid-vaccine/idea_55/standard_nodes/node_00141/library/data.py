import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# =========================================================================
# Mappings
# =========================================================================
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
# Loop types based on bpRNA: S: Stem, M: Multiloop, I: Internal loop,
# B: Bulge, H: Hairpin, E: dangling End, X: eXternal loop
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def get_structure_distance(structure: str) -> np.ndarray:
    """
    Calculates signed distance for paired bases to support Fixed Sinusoidal Encodings.
    If base i pairs with base j, then dist[i] = j - i.
    Unpaired bases have a distance of 0.
    """
    length = len(structure)
    dist = np.zeros(length, dtype=np.float32)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                start = stack.pop()
                end = i
                # Assign signed distance
                dist[start] = end - start
                dist[end] = start - end

    return dist


def process_dataframe(df: pd.DataFrame, mode: str = "train") -> dict:
    """
    Processes a dataframe into a dictionary of tensors.
    """
    sequences = []
    loops = []
    dists = []
    targets = []
    ids = []

    # Translation tables for fast mapping
    seq_trans = str.maketrans("AGCU", "0123")
    loop_trans = str.maketrans("SMIBHEX", "0123456")

    for _, row in df.iterrows():
        # 1. Sequence
        # Map A,G,C,U to 0,1,2,3
        seq_str = row["sequence"]
        seq_ints = [int(c) for c in seq_str.translate(seq_trans)]
        sequences.append(seq_ints)

        # 2. Loop Type
        # Map S,M,I,B,H,E,X to 0..6
        loop_str = row["predicted_loop_type"]
        loop_ints = [int(c) for c in loop_str.translate(loop_trans)]
        loops.append(loop_ints)

        # 3. Structure Distance
        struct_str = row["structure"]
        d = get_structure_distance(struct_str)
        dists.append(d)

        # 4. IDs
        ids.append(row["id"])

        # 5. Targets (only for train/val)
        if mode in ["train", "val"]:
            t_list = []
            for col in Config.TARGET_COLS:
                val = row[col]  # This is a list/array of length 68

                # Create full length vector (107)
                full_val = np.zeros(Config.SEQ_LEN, dtype=np.float32)

                # Fill the scored positions
                # Note: val length is usually 68 (Config.PRED_LEN)
                current_len = len(val)
                full_val[:current_len] = val

                t_list.append(full_val)

            # Stack to shape (107, 3)
            t_stack = np.stack(t_list, axis=1)
            targets.append(t_stack)

    # Convert lists to numpy arrays first for efficient tensor creation
    sequences = np.array(sequences, dtype=np.int64)
    loops = np.array(loops, dtype=np.int64)
    dists = np.array(dists, dtype=np.float32)

    if mode in ["train", "val"]:
        targets = np.array(targets, dtype=np.float32)
    else:
        targets = None

    return {
        "ids": ids,  # List of strings
        "sequence": torch.from_numpy(sequences),
        "loop_type": torch.from_numpy(loops),
        "pair_dist": torch.from_numpy(dists),
        "targets": torch.from_numpy(targets) if targets is not None else None,
    }


class RNADataset(Dataset):
    def __init__(self, data_dict, mode="train"):
        self.ids = data_dict["ids"]
        self.sequences = data_dict["sequence"]
        self.loop_types = data_dict["loop_type"]
        self.pair_dists = data_dict["pair_dist"]
        self.targets = data_dict["targets"]
        self.mode = mode

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample = {
            "sequence": self.sequences[idx],
            "loop_type": self.loop_types[idx],
            "pair_dist": self.pair_dists[idx],
        }

        if self.targets is not None:
            sample["targets"] = self.targets[idx]

        if self.mode == "test":
            sample["id"] = self.ids[idx]

        return sample


def save_cache_npz(data_dict, path):
    """
    Saves processed data to a .npz file to avoid pickle/torch.save.
    """
    save_dict = {}
    for mode in ["train", "val", "test"]:
        if mode not in data_dict:
            continue
        d = data_dict[mode]

        save_dict[f"{mode}_ids"] = np.array(d["ids"])
        save_dict[f"{mode}_seq"] = d["sequence"].numpy()
        save_dict[f"{mode}_loop"] = d["loop_type"].numpy()
        save_dict[f"{mode}_dist"] = d["pair_dist"].numpy()

        if d["targets"] is not None:
            save_dict[f"{mode}_targets"] = d["targets"].numpy()

    np.savez(path, **save_dict)


def load_cache_npz(path):
    """
    Loads processed data from a .npz file.
    """
    # allow_pickle=True is required to load the object array of ID strings
    # This is standard numpy behavior for string arrays and distinct from 'pickle' module usage
    data = np.load(path, allow_pickle=True)
    processed = {}

    for mode in ["train", "val", "test"]:
        if f"{mode}_seq" not in data:
            continue

        mode_dict = {}
        mode_dict["ids"] = data[f"{mode}_ids"].tolist()
        mode_dict["sequence"] = torch.from_numpy(data[f"{mode}_seq"])
        mode_dict["loop_type"] = torch.from_numpy(data[f"{mode}_loop"])
        mode_dict["pair_dist"] = torch.from_numpy(data[f"{mode}_dist"])

        if f"{mode}_targets" in data:
            mode_dict["targets"] = torch.from_numpy(data[f"{mode}_targets"])
        else:
            mode_dict["targets"] = None

        processed[mode] = mode_dict

    return processed


def get_dataloaders(load_cached_data=True, batch_size=None, debug_subset=None):
    """
    Creates DataLoaders for train, val, and test sets.
    Implements caching logic using .npz files.
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE

    cache_path = os.path.join(Config.CACHE_DIR, "processed_data.npz")
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    data_loaded = False
    processed_data = {}

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading cached data from {cache_path}...")
            processed_data = load_cache_npz(cache_path)
            data_loaded = True
        except Exception as e:
            print(f"Failed to load cache: {e}")
            data_loaded = False

    # 2. Process from scratch if needed
    if not data_loaded:
        print("Processing data from source Parquet files...")

        df_train = pd.read_parquet(Config.TRAIN_DATA_PATH)
        df_val = pd.read_parquet(Config.VAL_DATA_PATH)
        df_test = pd.read_parquet(Config.TEST_DATA_PATH)

        processed_data["train"] = process_dataframe(df_train, mode="train")
        processed_data["val"] = process_dataframe(df_val, mode="val")
        processed_data["test"] = process_dataframe(df_test, mode="test")

        print(f"Saving processed data to {cache_path}...")
        save_cache_npz(processed_data, cache_path)

    # 3. Handle Debugging (Subset)
    if debug_subset:
        print(f"Debugging: Slicing training set to {debug_subset} samples.")
        train_d = processed_data["train"]
        # Slice all arrays
        train_d["ids"] = train_d["ids"][:debug_subset]
        train_d["sequence"] = train_d["sequence"][:debug_subset]
        train_d["loop_type"] = train_d["loop_type"][:debug_subset]
        train_d["pair_dist"] = train_d["pair_dist"][:debug_subset]
        if train_d["targets"] is not None:
            train_d["targets"] = train_d["targets"][:debug_subset]

    # 4. Create Datasets
    train_dataset = RNADataset(processed_data["train"], mode="train")
    val_dataset = RNADataset(processed_data["val"], mode="val")
    test_dataset = RNADataset(processed_data["test"], mode="test")

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
