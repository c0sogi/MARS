import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# ==========================================
# Encoding Maps
# ==========================================
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {".": 0, "(": 1, ")": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]


def one_hot(seq, map_dict, length, num_classes):
    """
    One-hot encodes a sequence string based on a mapping dictionary.
    """
    res = np.zeros((length, num_classes), dtype=np.float32)
    for i, char in enumerate(seq):
        if i >= length:
            break
        if char in map_dict:
            res[i, map_dict[char]] = 1.0
    return res


def parse_structure_pairs(structure, length):
    """
    Parses dot-bracket structure to find base pairs.
    Returns:
        indices: Array where indices[i] = j if (i, j) are paired, else i.
        mask: Array where mask[i] = 1.0 if paired, else 0.0.
    """
    indices = np.arange(length, dtype=np.int64)
    mask = np.zeros(length, dtype=np.float32)
    stack = []

    for i, char in enumerate(structure):
        if i >= length:
            break
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


def process_dataframe(df, mode="train"):
    """
    Converts dataframe columns into numpy arrays for model input.
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LEN

    # Pre-allocate arrays
    # 4 (seq) + 3 (struct) + 7 (loop) = 14 channels
    inputs = np.zeros((num_samples, seq_len, 14), dtype=np.float32)
    bpp_indices = np.zeros((num_samples, seq_len), dtype=np.int64)
    bpp_masks = np.zeros((num_samples, seq_len), dtype=np.float32)
    targets = np.zeros((num_samples, seq_len, 5), dtype=np.float32)
    ids = []

    # Iterate over dataframe
    # Using itertuples for speed
    for idx, row in enumerate(df.itertuples(index=False)):
        # Extract features
        s_seq = getattr(row, "sequence")
        s_struct = getattr(row, "structure")
        s_loop = getattr(row, "predicted_loop_type")
        s_id = getattr(row, "id")

        ids.append(s_id)

        # One-hot encoding
        oh_seq = one_hot(s_seq, SEQ_MAP, seq_len, 4)
        oh_struct = one_hot(s_struct, STRUCT_MAP, seq_len, 3)
        oh_loop = one_hot(s_loop, LOOP_MAP, seq_len, 7)

        inputs[idx] = np.concatenate([oh_seq, oh_struct, oh_loop], axis=1)

        # Structure parsing
        p_indices, p_mask = parse_structure_pairs(s_struct, seq_len)
        bpp_indices[idx] = p_indices
        bpp_masks[idx] = p_mask

        # Target processing
        if mode in ["train", "val"]:
            # Targets are lists of length 68 (Config.PRED_LEN)
            # We pad them to 107 with zeros
            for t_i, col in enumerate(TARGET_COLS):
                val_list = getattr(row, col)
                # Ensure it's a list or array
                if isinstance(val_list, (list, np.ndarray)):
                    length = min(len(val_list), seq_len)
                    targets[idx, :length, t_i] = val_list[:length]

    return inputs, bpp_indices, bpp_masks, targets, np.array(ids)


class RNADataset(Dataset):
    def __init__(self, inputs, bpp_indices, bpp_masks, targets, ids):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.bpp_indices = torch.tensor(bpp_indices, dtype=torch.long)
        self.bpp_masks = torch.tensor(bpp_masks, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.ids = ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            "inputs": self.inputs[idx],
            "bpp_indices": self.bpp_indices[idx],
            "bpp_mask": self.bpp_masks[idx],
            "targets": self.targets[idx],
            "id": self.ids[idx],
        }


def get_loader(
    mode: str,
    batch_size: int = 32,
    num_workers: int = 4,
    load_cached_data: bool = True,
    shuffle: bool = None,
):
    """
    Prepares and returns a DataLoader for the specified mode.
    Handles caching logic to avoid re-processing data.

    Args:
        mode (str): 'train', 'val', or 'test'.
        batch_size (int): Batch size.
        num_workers (int): Number of worker processes.
        load_cached_data (bool): Whether to attempt loading from cache.
        shuffle (bool): Whether to shuffle the data. Defaults to True for train, False otherwise.
    """

    # Determine paths based on mode
    if mode == "train":
        meta_path = Config.TRAIN_METADATA
        cache_path = Config.TRAIN_CACHE
    elif mode == "val":
        meta_path = Config.VAL_METADATA
        cache_path = Config.VAL_CACHE
    elif mode == "test":
        meta_path = Config.TEST_METADATA
        cache_path = Config.TEST_CACHE
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Default shuffle logic
    if shuffle is None:
        shuffle = mode == "train"

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    data_loaded = False
    inputs, bpp_indices, bpp_masks, targets, ids = None, None, None, None, None

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading {mode} data from cache: {cache_path}")
            data = np.load(cache_path, allow_pickle=True)
            inputs = data["inputs"]
            bpp_indices = data["bpp_indices"]
            bpp_masks = data["bpp_masks"]
            targets = data["targets"]
            ids = data["ids"]
            data_loaded = True
        except Exception as e:
            print(f"Failed to load cache: {e}. Re-processing.")
            data_loaded = False

    # 2. Process from scratch if needed
    if not data_loaded:
        print(f"Processing {mode} data from metadata: {meta_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        df = pd.read_parquet(meta_path)
        inputs, bpp_indices, bpp_masks, targets, ids = process_dataframe(df, mode=mode)

        # Save to cache
        print(f"Saving {mode} data to cache: {cache_path}")
        np.savez(
            cache_path,
            inputs=inputs,
            bpp_indices=bpp_indices,
            bpp_masks=bpp_masks,
            targets=targets,
            ids=ids,
        )

    # 3. Create Dataset and DataLoader
    dataset = RNADataset(inputs, bpp_indices, bpp_masks, targets, ids)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=(
            mode == "train"
        ),  # Drop last batch in training to maintain batch norm stats stability
    )

    return loader
