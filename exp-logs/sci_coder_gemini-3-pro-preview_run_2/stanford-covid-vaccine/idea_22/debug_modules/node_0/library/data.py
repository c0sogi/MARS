import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed

# =========================================================================
# Mappings
# =========================================================================
SEQ_MAP = {"A": 0, "G": 1, "U": 2, "C": 3}
STRUCT_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def get_one_hot(seq, mapping, depth):
    """
    Converts a sequence string into a one-hot numpy array.
    """
    n = len(seq)
    one_hot = np.zeros((n, depth), dtype=np.float32)
    for i, char in enumerate(seq):
        if char in mapping:
            idx = mapping[char]
            one_hot[i, idx] = 1.0
    return one_hot


def get_pair_map(structure):
    """
    Parses dot-bracket structure to find pairs.
    Returns an array where arr[i] is the index of the partner of i.
    If i is unpaired, arr[i] = -1.
    """
    n = len(structure)
    pair_map = np.full(n, -1, dtype=np.int32)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                pair_map[i] = j
                pair_map[j] = i
    return pair_map


def parse_target_column(col_data, seq_len=107):
    """
    Parses a pandas Series of stringified lists into a padded numpy array.
    """
    # Parse strings to lists
    parsed = col_data.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

    # Convert to numpy array with padding
    # Targets are usually length 68, need to pad to 107
    num_samples = len(parsed)
    # Determine the length of the available data (usually 68)
    sample_len = (
        len(parsed.iloc[0]) if len(parsed) > 0 and len(parsed.iloc[0]) > 0 else 0
    )

    out = np.zeros((num_samples, seq_len), dtype=np.float32)

    if sample_len > 0:
        # Stack the lists
        # Note: This assumes all valid entries have the same length (68)
        # We handle potential empty lists or mismatches by iterating if necessary,
        # but for this dataset, vectorization is usually safe.
        # To be robust against empty lists in test set (though test set has no targets):
        valid_data = parsed.tolist()
        # Convert to array
        # Some rows might be empty or different length?
        # In this competition, train data is consistent.
        temp_arr = np.array(valid_data, dtype=np.float32)

        copy_len = min(seq_len, temp_arr.shape[1])
        out[:, :copy_len] = temp_arr[:, :copy_len]

    return out


class RNADataset(Dataset):
    def __init__(self, inputs, partner_indices, targets=None, ids=None):
        self.inputs = inputs
        self.partner_indices = partner_indices
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Inputs: (Seq_Len, Input_Dim)
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)

        # Partner Indices: (Seq_Len,)
        p_idx = torch.tensor(self.partner_indices[idx], dtype=torch.long)

        sample = {"inputs": x, "partner_indices": p_idx}

        if self.ids is not None:
            sample["id"] = self.ids[idx]

        if self.targets is not None:
            # Targets: (Seq_Len, Num_Targets)
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            sample["targets"] = y

        return sample


def process_and_cache_data(csv_path, cache_path, is_test=False, load_cached_data=True):
    """
    Loads data from CSV, processes features, and caches to NPZ.
    If cache exists and load_cached_data is True, loads from cache.
    """
    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            data = np.load(cache_path, allow_pickle=True)
            inputs = data["inputs"]
            partner_indices = data["partner_indices"]
            ids = data["ids"]
            targets = data["targets"] if "targets" in data else None
            return inputs, partner_indices, targets, ids
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Load CSV
    print(f"Processing data from {csv_path}...")
    df = pd.read_csv(csv_path)

    num_samples = len(df)
    seq_len = Config.SEQ_LEN

    # 3. Pre-allocate arrays
    # Input Dim: 4 (Seq) + 3 (Struct) + 7 (Loop) + 4 (PartnerID) = 18
    input_dim = Config.INPUT_DIM

    inputs = np.zeros((num_samples, seq_len, input_dim), dtype=np.float32)
    partner_indices_arr = np.zeros((num_samples, seq_len), dtype=np.int32)

    # 4. Feature Generation Loop
    # Iterating is fast enough for ~2k samples
    for idx, row in df.iterrows():
        seq = row["sequence"]
        struct = row["structure"]
        loop = row["predicted_loop_type"]

        # Basic One-Hots
        o_seq = get_one_hot(seq, SEQ_MAP, 4)
        o_struct = get_one_hot(struct, STRUCT_MAP, 3)
        o_loop = get_one_hot(loop, LOOP_MAP, 7)

        # Partner Mapping
        # pair_map has -1 for unpaired
        pair_map = get_pair_map(struct)

        # Partner Identity Feature
        # If paired (j != -1), use one_hot(seq[j]). Else zeros.
        o_partner = np.zeros((seq_len, 4), dtype=np.float32)
        valid_pairs = pair_map != -1
        # Indices of partners
        partners = pair_map[valid_pairs]
        # Indices of current bases
        currents = np.where(valid_pairs)[0]

        # Assign partner features
        # We can use the already computed o_seq
        o_partner[currents] = o_seq[partners]

        # Concatenate Features
        # [Seq, Struct, Loop, PartnerID]
        feat = np.concatenate([o_seq, o_struct, o_loop, o_partner], axis=1)
        inputs[idx] = feat

        # Partner Indices for Model Gather
        # The model needs valid indices for gather.
        # Map unpaired (-1) to self (i) so gather retrieves self-features (harmless/neutral).
        # Or we could map to 0, but self is semantically cleaner for "local" context.
        gather_indices = pair_map.copy()
        gather_indices[gather_indices == -1] = np.arange(seq_len)[gather_indices == -1]
        partner_indices_arr[idx] = gather_indices

    # 5. Target Processing
    targets = None
    if not is_test:
        target_cols = (
            Config.TARGET_COLS
        )  # ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
        target_arrays = []
        for col in target_cols:
            # Parse and pad
            arr = parse_target_column(df[col], seq_len=seq_len)
            target_arrays.append(arr)

        # Stack to (N, Seq_Len, 5)
        targets = np.stack(target_arrays, axis=2)

    # 6. Save to Cache
    ids = df["id"].values
    print(f"Saving processed data to {cache_path}...")
    save_dict = {"inputs": inputs, "partner_indices": partner_indices_arr, "ids": ids}
    if targets is not None:
        save_dict["targets"] = targets

    np.savez_compressed(cache_path, **save_dict)

    return inputs, partner_indices_arr, targets, ids


def get_loader(
    split="train",
    batch_size=Config.BATCH_SIZE,
    shuffle=True,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
):
    """
    Creates a DataLoader for the specified split.

    Args:
        split (str): 'train', 'val', or 'test'.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of worker processes.
        load_cached_data (bool): Whether to use cached .npz files.

    Returns:
        DataLoader: PyTorch DataLoader.
    """
    # Determine paths based on split
    if split == "train":
        csv_path = Config.TRAIN_CSV
        cache_path = Config.TRAIN_CACHE
        is_test = False
    elif split == "val":
        csv_path = Config.VAL_CSV
        cache_path = Config.VAL_CACHE
        is_test = False
    elif split == "test":
        csv_path = Config.TEST_CSV
        cache_path = Config.TEST_CACHE
        is_test = True
    else:
        raise ValueError(f"Unknown split: {split}")

    # Ensure working directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Process Data
    inputs, partner_indices, targets, ids = process_and_cache_data(
        csv_path, cache_path, is_test=is_test, load_cached_data=load_cached_data
    )

    # Create Dataset
    dataset = RNADataset(inputs, partner_indices, targets, ids)

    # Create Loader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    return loader
