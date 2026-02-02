import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# =========================================================================
# Feature Mappings
# =========================================================================
TOKEN2INT_SEQ = {x: i for i, x in enumerate("AGUC")}
TOKEN2INT_STRUCT = {x: i for i, x in enumerate("().")}
TOKEN2INT_LOOP = {x: i for i, x in enumerate("SMIBHEX")}


def get_partners(structure):
    """
    Parses a dot-bracket structure string to find base pairs.
    Returns an array where arr[i] is the index of the partner of base i.
    If base i is unpaired, arr[i] = i (points to self) to ensure valid indexing
    during gathering operations.
    """
    partners = np.arange(len(structure), dtype=np.int32)
    stack = []
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                partners[i] = j
                partners[j] = i
    return partners


def preprocess_data(csv_path, mode="train", load_cached_data=True):
    """
    Preprocesses the data from the CSV file.
    Generates inputs (one-hot seq, struct, loop, partner_id), partner indices, and targets.
    Handles caching using .npz files to save time on subsequent runs.
    """
    # Construct cache path based on filename and version
    fname = os.path.basename(csv_path).replace(".csv", "")
    cache_path = os.path.join(
        Config.WORKING_DIR, f"{fname}_data_{Config.CACHE_VERSION}.npz"
    )

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            data = np.load(cache_path, allow_pickle=True)
            # Validate that required keys exist in the cache
            required_keys = ["inputs", "partner_indices", "targets", "ids"]
            if not all(key in data for key in required_keys):
                raise ValueError(
                    f"Cache file missing required keys. Found: {list(data.keys())}"
                )
            return dict(data)
        except Exception as e:
            print(f"Failed to load cache: {e}. Regenerating...")

    # 2. Generate data from scratch
    print(f"Preprocessing data from {csv_path}...")
    df = pd.read_csv(csv_path)

    num_samples = len(df)
    seq_len = Config.SEQ_LENGTH

    # Input features: Seq(4) + Struct(3) + Loop(7) + PartnerID(4) = 18
    input_array = np.zeros((num_samples, seq_len, 18), dtype=np.float32)
    partner_index_array = np.zeros((num_samples, seq_len), dtype=np.int64)

    # Targets
    # For train/val, we have targets of length 68.
    # For test, we create placeholders of length 107 (full sequence).
    if mode in ["train", "val"]:
        target_len = Config.PRED_LEN
        target_array = np.zeros((num_samples, target_len, 5), dtype=np.float32)
    else:
        target_array = np.zeros((num_samples, seq_len, 5), dtype=np.float32)

    ids = []

    for idx, row in df.iterrows():
        sequence = row["sequence"]
        structure = row["structure"]
        loop_type = row["predicted_loop_type"]

        # --- 1. Basic One-Hot Encoding ---
        # Sequence (4)
        seq_int = [TOKEN2INT_SEQ.get(x, 0) for x in sequence]
        seq_oh = np.eye(4)[seq_int]

        # Structure (3)
        struct_int = [TOKEN2INT_STRUCT.get(x, 2) for x in structure]  # Default to .
        struct_oh = np.eye(3)[struct_int]

        # Loop Type (7)
        loop_int = [TOKEN2INT_LOOP.get(x, 6) for x in loop_type]  # Default to X
        loop_oh = np.eye(7)[loop_int]

        # --- 2. Partner Info ---
        partners = get_partners(structure)  # (L,) indices
        partner_index_array[idx] = partners

        # Partner Identity (4)
        # If paired (i, j), partner_id at i is one-hot of sequence[j]
        # If unpaired (i, i), we mask it to zero vector to represent 'no partner'
        is_paired = partners != np.arange(len(sequence))

        # Gather partner sequence indices
        partner_seq_int = np.array([seq_int[p] for p in partners])
        partner_id_oh = np.eye(4)[partner_seq_int]

        # Mask unpaired positions to 0 vector
        partner_id_oh[~is_paired] = 0.0

        # Concatenate all inputs
        # Shape: (L, 4+3+7+4) = (L, 18)
        full_input = np.concatenate([seq_oh, struct_oh, loop_oh, partner_id_oh], axis=1)
        input_array[idx] = full_input

        # --- 3. Targets ---
        if mode in ["train", "val"]:
            # Columns: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
            t_vals = []
            for col in Config.TARGET_COLS:
                val_str = row[col]
                try:
                    val_list = ast.literal_eval(val_str)
                except:
                    # Fallback if parsing fails (should not happen with clean metadata)
                    val_list = [0.0] * Config.PRED_LEN
                t_vals.append(val_list)

            # Stack to (68, 5)
            # t_vals is list of 5 lists of length 68 -> (5, 68) -> transpose to (68, 5)
            t_vals = np.array(t_vals, dtype=np.float32).T
            target_array[idx] = t_vals

        ids.append(row["id"])

    # Save to cache
    # We use np.savez to store multiple arrays.
    # ids are stored as numpy array of strings.
    np.savez(
        cache_path,
        inputs=input_array,
        partner_indices=partner_index_array,
        targets=target_array,
        ids=np.array(ids),
    )

    print(f"Data processed and saved to {cache_path}")

    return {
        "inputs": input_array,
        "partner_indices": partner_index_array,
        "targets": target_array,
        "ids": np.array(ids),
    }


class RNADataset(Dataset):
    def __init__(self, data_dict):
        self.inputs = data_dict["inputs"]
        self.partner_indices = data_dict["partner_indices"]
        self.targets = data_dict["targets"]
        self.ids = data_dict["ids"]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Convert to torch tensors
        # Inputs: (SeqLen, 18)
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)
        # Partner Indices: (SeqLen,)
        p_idx = torch.tensor(self.partner_indices[idx], dtype=torch.long)
        # Targets: (PredLen, 5) or (SeqLen, 5)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)

        return x, p_idx, y


def get_loader(
    mode, batch_size=32, num_workers=2, load_cached_data=True, limit_size=None
):
    """
    Creates a DataLoader for the specified mode.

    Args:
        mode (str): 'train', 'val', or 'test'.
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.
        load_cached_data (bool): Whether to use cached .npz files.
        limit_size (int, optional): If set, limits the dataset size (useful for debugging).
    """
    if mode == "train":
        csv_path = Config.TRAIN_CSV
        shuffle = True
    elif mode == "val":
        csv_path = Config.VAL_CSV
        shuffle = False
    elif mode == "test":
        csv_path = Config.TEST_CSV
        shuffle = False
    else:
        raise ValueError(f"Unknown mode: {mode}")

    data_dict = preprocess_data(csv_path, mode=mode, load_cached_data=load_cached_data)

    # Apply limit if requested
    if limit_size is not None:
        print(f"Limiting dataset to first {limit_size} samples.")
        data_dict = {k: v[:limit_size] for k, v in data_dict.items()}

    dataset = RNADataset(data_dict)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return loader
