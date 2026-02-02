import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed

# =========================================================================
# Constants & Mappings
# =========================================================================
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA Degradation Prediction.

    Returns:
        dict:
            - features: (107, 14) Float Tensor (Seq + Struct + Loop OneHot)
            - pair_indices: (107,) Long Tensor (Index of paired base, or -1)
            - unpaired_mask: (107,) Bool Tensor (True if unpaired)
            - targets: (107, 5) Float Tensor (Ground truth, optional)
            - ids: String ID of the sample
    """

    def __init__(self, features, pair_indices, unpaired_mask, targets=None, ids=None):
        self.features = features
        self.pair_indices = pair_indices
        self.unpaired_mask = unpaired_mask
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        item = {
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
            "pair_indices": torch.tensor(self.pair_indices[idx], dtype=torch.long),
            "unpaired_mask": torch.tensor(self.unpaired_mask[idx], dtype=torch.bool),
        }

        if self.targets is not None:
            item["targets"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        if self.ids is not None:
            item["ids"] = self.ids[idx]

        return item


def parse_structure(structure_str):
    """
    Parses dot-bracket structure to find pair indices.

    Args:
        structure_str (str): Dot-bracket string e.g. "((..))"

    Returns:
        np.array: Array of shape (L,) where indices[i] is the index of the pair.
                  If i is unpaired, indices[i] = -1.
    """
    n = len(structure_str)
    indices = np.full(n, -1, dtype=np.int32)
    stack = []

    for i, char in enumerate(structure_str):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                indices[i] = j
                indices[j] = i
    return indices


def encode_sequence(seq, mapping, length):
    """
    One-hot encodes a sequence string based on a mapping dictionary.
    """
    arr = np.zeros((length, len(mapping)), dtype=np.float32)
    for i, char in enumerate(seq):
        if i >= length:
            break
        if char in mapping:
            arr[i, mapping[char]] = 1.0
    return arr


def preprocess_data(df, mode="train"):
    """
    Converts DataFrame to numpy arrays for features and targets.
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LEN

    # Initialize arrays
    # Features: (N, L, 14) -> 4 (Seq) + 3 (Struct) + 7 (Loop)
    features = np.zeros((num_samples, seq_len, 14), dtype=np.float32)
    pair_indices = np.zeros((num_samples, seq_len), dtype=np.int32)
    unpaired_mask = np.zeros((num_samples, seq_len), dtype=bool)

    # Targets: (N, L, 5) - Only for train/val
    targets = None
    if mode in ["train", "val"]:
        targets = np.zeros((num_samples, seq_len, 5), dtype=np.float32)

    ids = df["id"].values

    for idx, row in df.iterrows():
        # 1. Features
        seq_oh = encode_sequence(row["sequence"], SEQ_MAP, seq_len)
        struct_oh = encode_sequence(row["structure"], STRUCT_MAP, seq_len)
        loop_oh = encode_sequence(row["predicted_loop_type"], LOOP_MAP, seq_len)

        features[idx] = np.concatenate([seq_oh, struct_oh, loop_oh], axis=1)

        # 2. Pair Indices & Mask
        p_idx = parse_structure(row["structure"])
        pair_indices[idx] = p_idx
        unpaired_mask[idx] = p_idx == -1

        # 3. Targets
        if mode in ["train", "val"]:
            # Targets are lists of length 68 (Config.PRED_LEN)
            # We pad them to 107 with zeros (loss function handles slicing)
            for t_i, col in enumerate(Config.TARGET_COLS):
                val_list = row[col]
                if isinstance(val_list, (list, np.ndarray)):
                    length = len(val_list)
                    targets[idx, :length, t_i] = val_list

    return {
        "features": features,
        "pair_indices": pair_indices,
        "unpaired_mask": unpaired_mask,
        "targets": targets,
        "ids": ids,
    }


def get_dataloaders(load_cached_data=True, subset_size=None):
    """
    Factory function to create DataLoaders for train, val, and test.

    Args:
        load_cached_data (bool): If True, attempts to load preprocessed .npz files.
        subset_size (int, optional): If set, limits dataset size for debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    set_seed(Config.SEED)

    # Ensure working directory exists for caching
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    splits = ["train", "val", "test"]
    datasets = {}

    for split in splits:
        cache_path = os.path.join(Config.WORKING_DIR, f"{split}_data.npz")
        data_dict = None

        # 1. Try Loading Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading {split} data from cache: {cache_path}")
            try:
                loaded = np.load(cache_path, allow_pickle=True)
                data_dict = {
                    "features": loaded["features"],
                    "pair_indices": loaded["pair_indices"],
                    "unpaired_mask": loaded["unpaired_mask"],
                    "ids": loaded["ids"],
                }
                # Check if targets exist in cache
                if "targets" in loaded:
                    data_dict["targets"] = loaded["targets"]
                else:
                    data_dict["targets"] = None
            except Exception as e:
                print(f"Failed to load cache for {split}: {e}")
                data_dict = None

        # 2. Process from Metadata if Cache Failed or Disabled
        if data_dict is None:
            print(f"Processing {split} data from metadata...")
            meta_path = os.path.join(Config.METADATA_DIR, f"{split}.parquet")

            if not os.path.exists(meta_path):
                raise FileNotFoundError(f"Metadata file not found: {meta_path}")

            df = pd.read_parquet(meta_path)
            data_dict = preprocess_data(df, mode=split)

            # Save to cache
            save_dict = {
                "features": data_dict["features"],
                "pair_indices": data_dict["pair_indices"],
                "unpaired_mask": data_dict["unpaired_mask"],
                "ids": data_dict["ids"],
            }
            if data_dict["targets"] is not None:
                save_dict["targets"] = data_dict["targets"]

            np.savez(cache_path, **save_dict)
            print(f"Saved {split} data to cache.")

        # 3. Apply Subset (Debugging)
        if subset_size is not None:
            print(f"Subsetting {split} to {subset_size} samples.")
            limit = min(subset_size, len(data_dict["features"]))
            data_dict["features"] = data_dict["features"][:limit]
            data_dict["pair_indices"] = data_dict["pair_indices"][:limit]
            data_dict["unpaired_mask"] = data_dict["unpaired_mask"][:limit]
            data_dict["ids"] = data_dict["ids"][:limit]
            if data_dict["targets"] is not None:
                data_dict["targets"] = data_dict["targets"][:limit]

        # 4. Create Dataset
        datasets[split] = RNADataset(
            features=data_dict["features"],
            pair_indices=data_dict["pair_indices"],
            unpaired_mask=data_dict["unpaired_mask"],
            targets=data_dict["targets"],
            ids=data_dict["ids"],
        )

    # 5. Create DataLoaders
    train_loader = DataLoader(
        datasets["train"],
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
        drop_last=True,
    )

    val_loader = DataLoader(
        datasets["val"],
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    test_loader = DataLoader(
        datasets["test"],
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    return train_loader, val_loader, test_loader
