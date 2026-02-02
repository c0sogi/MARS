import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config

# Mappings
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {".": 0, "(": 1, ")": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def get_partner_map(structure):
    """
    Parses dot-bracket structure to find paired indices.
    Returns a numpy array of shape (L,) where arr[i] is the index of the partner
    of base i, or -1 if unpaired.
    """
    length = len(structure)
    partner_map = np.full(length, -1, dtype=np.int32)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                partner_map[i] = j
                partner_map[j] = i

    return partner_map


def get_one_hot(indices, vocab_size):
    """
    Converts integer indices to one-hot encoding.
    Shape: (L, vocab_size)
    """
    res = np.zeros((len(indices), vocab_size), dtype=np.float32)
    valid_mask = (indices >= 0) & (indices < vocab_size)
    res[np.arange(len(indices))[valid_mask], indices[valid_mask]] = 1.0
    return res


def get_partner_identity(seq_one_hot, partner_map):
    """
    Generates one-hot encoding of the partner base.
    If a base is unpaired (partner_map == -1), returns a zero vector.
    """
    length, vocab_size = seq_one_hot.shape
    partner_identity = np.zeros((length, vocab_size), dtype=np.float32)

    # Indices where a partner exists
    paired_indices = np.where(partner_map != -1)[0]
    partners = partner_map[paired_indices]

    # Gather features from the partner positions
    partner_identity[paired_indices] = seq_one_hot[partners]

    return partner_identity


def process_data(data_type="train", load_cached_data=True, debug=False):
    """
    Loads, processes, and caches data.

    Args:
        data_type (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to load from cache if available.
        debug (bool): If True, processes a small subset.

    Returns:
        tuple: (inputs, partner_map, targets, ids)
            inputs: (N, 107, 18) - Concatenated features
            partner_map: (N, 107) - Indices of paired bases
            targets: (N, 68, 5) or None
            ids: List of IDs
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    cache_filename = f"{data_type}_data_{Config.CACHE_KEY}.npz"
    if debug:
        cache_filename = f"debug_{cache_filename}"

    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {data_type} data from cache: {cache_path}")
        try:
            data = np.load(cache_path, allow_pickle=True)
            inputs = data["inputs"]
            partner_map = data["partner_map"]
            ids = data["ids"]
            targets = data["targets"] if "targets" in data else None

            # If targets was saved as None (object array with None), handle it
            if targets is not None and targets.shape == ():
                targets = None

            return inputs, partner_map, targets, ids
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from scratch
    print(f"Processing {data_type} data from source...")

    if data_type == "train":
        csv_path = Config.TRAIN_CSV
    elif data_type == "val":
        csv_path = Config.VAL_CSV
    elif data_type == "test":
        csv_path = Config.TEST_CSV
    else:
        raise ValueError(f"Unknown data_type: {data_type}")

    df = pd.read_csv(csv_path)

    if debug:
        df = df.head(Config.DEBUG_SUBSET_SIZE)

    # Pre-allocate lists
    all_inputs = []
    all_partner_maps = []
    all_targets = []
    all_ids = df["id"].tolist()

    for idx, row in df.iterrows():
        # --- Feature Engineering ---
        seq_str = row["sequence"]
        struct_str = row["structure"]
        loop_str = row["predicted_loop_type"]

        # Convert strings to indices
        seq_idx = np.array([SEQ_MAP.get(c, 0) for c in seq_str])
        struct_idx = np.array([STRUCT_MAP.get(c, 0) for c in struct_str])
        loop_idx = np.array([LOOP_MAP.get(c, 0) for c in loop_str])

        # Get One-Hots
        seq_oh = get_one_hot(seq_idx, Config.VOCAB_SIZE_SEQ)  # (107, 4)
        struct_oh = get_one_hot(struct_idx, Config.VOCAB_SIZE_STRUCT)  # (107, 3)
        loop_oh = get_one_hot(loop_idx, Config.VOCAB_SIZE_LOOP)  # (107, 7)

        # Partner Map & Identity
        p_map = get_partner_map(struct_str)
        partner_identity = get_partner_identity(seq_oh, p_map)  # (107, 4)

        # Concatenate Features: 4 + 3 + 7 + 4 = 18 channels
        # Branch A (Identity) and Branch B (Context) will be handled in the model,
        # but we provide all raw features here.
        sample_input = np.concatenate(
            [seq_oh, struct_oh, loop_oh, partner_identity], axis=1
        )

        all_inputs.append(sample_input)
        all_partner_maps.append(p_map)

        # --- Target Parsing ---
        if data_type in ["train", "val"]:
            # Parse targets
            t_list = []
            for col in Config.TARGET_COLS:
                # Parse string representation of list
                val_list = ast.literal_eval(row[col])
                t_list.append(val_list)

            # Shape: (5, 68) -> Transpose to (68, 5)
            sample_targets = np.array(t_list, dtype=np.float32).T
            all_targets.append(sample_targets)

    # Convert to numpy arrays
    inputs_arr = np.array(all_inputs, dtype=np.float32)  # (N, 107, 18)
    partner_map_arr = np.array(all_partner_maps, dtype=np.int32)  # (N, 107)

    if data_type in ["train", "val"]:
        targets_arr = np.array(all_targets, dtype=np.float32)  # (N, 68, 5)
    else:
        targets_arr = None

    # 3. Save to cache
    print(f"Saving processed data to {cache_path}...")
    save_dict = {"inputs": inputs_arr, "partner_map": partner_map_arr, "ids": all_ids}
    if targets_arr is not None:
        save_dict["targets"] = targets_arr

    np.savez_compressed(cache_path, **save_dict)

    return inputs_arr, partner_map_arr, targets_arr, all_ids


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA degradation prediction.
    Yields:
        inputs: (107, 18)
        partner_map: (107,)
        targets: (68, 5) (if available)
    """

    def __init__(self, inputs, partner_map, targets=None):
        self.inputs = inputs
        self.partner_map = partner_map
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Convert to tensors
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)
        p_map = torch.tensor(self.partner_map[idx], dtype=torch.long)

        if self.targets is not None:
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            return x, p_map, y
        else:
            return x, p_map
