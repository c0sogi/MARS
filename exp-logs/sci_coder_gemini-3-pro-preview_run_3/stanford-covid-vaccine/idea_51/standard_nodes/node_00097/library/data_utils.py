import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# =============================================================================
# Constants & Mappings
# =============================================================================

TOKEN2INT_SEQ = {x: i for i, x in enumerate("AGCU")}
TOKEN2INT_STRUCT = {x: i for i, x in enumerate("().")}
TOKEN2INT_LOOP = {x: i for i, x in enumerate("SMIBHEX")}


def seed_everything(seed):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# =============================================================================
# Helper Functions
# =============================================================================


def get_bpp_indices(structure):
    """
    Parses dot-bracket structure to generate adjacency indices and masks.

    Args:
        structure (str): Dot-bracket string e.g., "((..))"

    Returns:
        indices (np.array): Array of shape (L,) where indices[i] = j if i pairs with j.
                            If unpaired, indices[i] = 0 (to be valid index).
        mask (np.array): Array of shape (L,) where mask[i] = 1 if paired, 0 if unpaired.
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

    return indices, mask


def one_hot_encode(seq, struct, loop):
    """
    Generates (L, 14) one-hot encoded tensor.
    Channels: 4 (Seq) + 3 (Struct) + 7 (Loop)
    """
    length = len(seq)
    encoding = np.zeros((length, 14), dtype=np.float32)

    for i in range(length):
        # Sequence (0-3)
        if seq[i] in TOKEN2INT_SEQ:
            encoding[i, TOKEN2INT_SEQ[seq[i]]] = 1.0

        # Structure (4-6)
        if struct[i] in TOKEN2INT_STRUCT:
            encoding[i, 4 + TOKEN2INT_STRUCT[struct[i]]] = 1.0

        # Loop Type (7-13)
        if loop[i] in TOKEN2INT_LOOP:
            encoding[i, 7 + TOKEN2INT_LOOP[loop[i]]] = 1.0

    return encoding


def preprocess_data(df, has_targets=True):
    """
    Converts DataFrame into numpy arrays for model consumption.
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LEN

    # Inputs
    inputs = np.zeros((num_samples, seq_len, Config.INPUT_CHANNELS), dtype=np.float32)
    bpp_indices = np.zeros((num_samples, seq_len), dtype=np.int64)
    bpp_masks = np.zeros((num_samples, seq_len), dtype=np.float32)
    ids = []

    # Targets (only if has_targets)
    # 5 targets * 107 positions (padded from 68)
    targets = None
    target_masks = None

    if has_targets:
        targets = np.zeros((num_samples, seq_len, 5), dtype=np.float32)
        target_masks = np.zeros((num_samples, seq_len), dtype=np.float32)

    for idx, row in df.iterrows():
        # 1. Process Inputs
        seq = row["sequence"]
        struct = row["structure"]
        loop = row["predicted_loop_type"]

        # Safety check for length
        if len(seq) != seq_len:
            # In case of length mismatch, we truncate or pad (though dataset spec says 107)
            # For this competition data, it is consistently 107.
            pass

        inputs[idx] = one_hot_encode(seq, struct, loop)
        b_idx, b_mask = get_bpp_indices(struct)
        bpp_indices[idx] = b_idx
        bpp_masks[idx] = b_mask
        ids.append(row["id"])

        # 2. Process Targets
        if has_targets:
            # Target columns are lists of length 68
            # We pad them to 107
            scored_len = row["seq_scored"]

            for t_i, col in enumerate(Config.TARGET_COLS):
                val_list = row[col]
                # Ensure it's a list or array
                if isinstance(val_list, (list, np.ndarray)):
                    length = min(len(val_list), seq_len)
                    targets[idx, :length, t_i] = val_list[:length]

            # Create mask for scored positions
            target_masks[idx, :scored_len] = 1.0

    data_dict = {
        "inputs": inputs,
        "bpp_indices": bpp_indices,
        "bpp_masks": bpp_masks,
        "ids": np.array(ids),
    }

    if has_targets:
        data_dict["targets"] = targets
        data_dict["target_masks"] = target_masks

    return data_dict


def load_or_process_data(path, cache_path, has_targets=True, load_cached_data=True):
    """
    Loads data from cache if available, otherwise processes from Parquet.
    """
    # Ensure working directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            return np.load(cache_path, allow_pickle=True).item()
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    print(f"Processing data from {path}...")
    df = pd.read_parquet(path)
    data = preprocess_data(df, has_targets=has_targets)

    print(f"Saving cache to {cache_path}...")
    np.save(cache_path, data)

    return data


# =============================================================================
# Dataset Class
# =============================================================================


class RNADataset(Dataset):
    def __init__(self, data_dict, mode="train"):
        self.inputs = data_dict["inputs"]
        self.bpp_indices = data_dict["bpp_indices"]
        self.bpp_masks = data_dict["bpp_masks"]
        self.ids = data_dict["ids"]
        self.mode = mode

        if mode in ["train", "val"]:
            self.targets = data_dict["targets"]
            self.target_masks = data_dict["target_masks"]
        else:
            self.targets = None
            self.target_masks = None

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        item = {
            "inputs": torch.tensor(self.inputs[idx], dtype=torch.float32),
            "bpp_indices": torch.tensor(self.bpp_indices[idx], dtype=torch.long),
            "bpp_masks": torch.tensor(self.bpp_masks[idx], dtype=torch.float32),
            "ids": self.ids[idx],
        }

        if self.targets is not None:
            item["targets"] = torch.tensor(self.targets[idx], dtype=torch.float32)
            item["target_masks"] = torch.tensor(
                self.target_masks[idx], dtype=torch.float32
            )

        return item


# =============================================================================
# DataLoader Factory
# =============================================================================


def get_dataloaders(load_cached_data=True):
    """
    Main entry point to get DataLoaders for Train, Val, and Test.
    """
    seed_everything(Config.SEED)

    # 1. Load Data
    train_data = load_or_process_data(
        Config.TRAIN_PATH,
        Config.TRAIN_CACHE,
        has_targets=True,
        load_cached_data=load_cached_data,
    )

    val_data = load_or_process_data(
        Config.VAL_PATH,
        Config.VAL_CACHE,
        has_targets=True,
        load_cached_data=load_cached_data,
    )

    test_data = load_or_process_data(
        Config.TEST_PATH,
        Config.TEST_CACHE,
        has_targets=False,
        load_cached_data=load_cached_data,
    )

    # 2. Create Datasets
    train_dataset = RNADataset(train_data, mode="train")
    val_dataset = RNADataset(val_data, mode="val")
    test_dataset = RNADataset(test_data, mode="test")

    # 3. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
