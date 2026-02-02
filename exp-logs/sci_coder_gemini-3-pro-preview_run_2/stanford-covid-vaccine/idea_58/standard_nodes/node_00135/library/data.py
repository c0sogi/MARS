import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


def get_couples(structure):
    """
    Converts dot-bracket structure to a list of paired indices.
    Returns a mapping where map[i] = j if i is paired with j, else -1.
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


def process_data(df):
    """
    Generates inputs and targets from the dataframe.

    Features (18 channels):
    - 0-3: Sequence (A, G, C, U)
    - 4-6: Structure (., (, ))
    - 7-13: Loop Type (S, M, I, B, H, E, X)
    - 14-17: Partner Base Identity (A, G, C, U of the paired base)
    """
    ids = df["id"].values
    sequences = df["sequence"].values
    structures = df["structure"].values
    loops = df["predicted_loop_type"].values

    n_samples = len(df)
    seq_len = Config.SEQ_LEN

    # Mappings
    base_map = {c: i for i, c in enumerate(Config.BASES)}
    struct_map = {c: i for i, c in enumerate(Config.STRUCTS)}
    loop_map = {c: i for i, c in enumerate(Config.LOOPS)}

    # Pre-allocate
    features = np.zeros((n_samples, Config.INPUT_DIM, seq_len), dtype=np.float32)
    bpp_indices = np.full((n_samples, seq_len), -1, dtype=np.int32)

    for idx in range(n_samples):
        seq = sequences[idx]
        struct = structures[idx]
        loop = loops[idx]

        # Parse pairs
        pairs = get_couples(struct)
        bpp_indices[idx] = pairs

        for i in range(seq_len):
            # One-Hot Sequence (0-3)
            if seq[i] in base_map:
                features[idx, base_map[seq[i]], i] = 1.0

            # One-Hot Structure (4-6)
            if struct[i] in struct_map:
                features[idx, 4 + struct_map[struct[i]], i] = 1.0

            # One-Hot Loop (7-13)
            if loop[i] in loop_map:
                features[idx, 7 + loop_map[loop[i]], i] = 1.0

            # Partner Identity (14-17)
            partner_idx = pairs[i]
            if partner_idx != -1:
                partner_base = seq[partner_idx]
                if partner_base in base_map:
                    features[idx, 14 + base_map[partner_base], i] = 1.0

    # Parse Targets (if available)
    targets = None
    # Check if target columns exist
    if Config.TARGET_COLS[0] in df.columns:
        targets = np.zeros(
            (n_samples, seq_len, len(Config.TARGET_COLS)), dtype=np.float32
        )

        for t_i, col in enumerate(Config.TARGET_COLS):
            # Helper to safely parse stringified lists
            def parse_val(x):
                if isinstance(x, str):
                    try:
                        return ast.literal_eval(x)
                    except:
                        return [0.0] * seq_len
                elif isinstance(x, (list, np.ndarray)):
                    return x
                return [0.0] * seq_len

            values = df[col].apply(parse_val).tolist()

            for idx, val_list in enumerate(values):
                length = min(len(val_list), seq_len)
                targets[idx, :length, t_i] = val_list[:length]

    return {
        "ids": ids,
        "features": features,
        "bpp_indices": bpp_indices,
        "targets": targets,
    }


def load_dataset(mode="train", load_cached_data=True):
    """
    Loads data with caching mechanism.
    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to try loading from cache.
    """
    cache_path = getattr(Config, f"CACHE_{mode.upper()}")
    csv_path = getattr(Config, f"{mode.upper()}_CSV")

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {mode} data from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return {
            "ids": data["ids"],
            "features": data["features"],
            "bpp_indices": data["bpp_indices"],
            "targets": data["targets"] if "targets" in data else None,
        }

    # 2. Process from Scratch
    print(f"Processing {mode} data from {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found.")

    df = pd.read_csv(csv_path)

    # Debug subset
    if Config.DEBUG and mode == "train":
        df = df.head(Config.MAX_DEBUG_SAMPLES)

    processed = process_data(df)

    # 3. Save Cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    save_dict = {
        "ids": processed["ids"],
        "features": processed["features"],
        "bpp_indices": processed["bpp_indices"],
    }
    if processed["targets"] is not None:
        save_dict["targets"] = processed["targets"]

    np.savez_compressed(cache_path, **save_dict)
    print(f"Saved cache to {cache_path}")

    return processed


class RNADataset(Dataset):
    def __init__(self, data, mode="train"):
        self.features = data["features"]
        self.bpp_indices = data["bpp_indices"]
        self.targets = data["targets"]
        self.mode = mode

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Features: (18, 107)
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        # BPP: (107,)
        bpp = torch.tensor(self.bpp_indices[idx], dtype=torch.long)

        if self.targets is not None:
            # Targets: (107, 5)
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            return x, bpp, y
        else:
            return x, bpp


def get_dataloaders(batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS):
    """
    Creates DataLoaders for train, val, and test sets.
    """
    # Load Data
    train_data = load_dataset("train", load_cached_data=True)
    val_data = load_dataset("val", load_cached_data=True)
    test_data = load_dataset("test", load_cached_data=True)

    # Create Datasets
    train_ds = RNADataset(train_data, mode="train")
    val_ds = RNADataset(val_data, mode="val")
    test_ds = RNADataset(test_data, mode="test")

    # Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
