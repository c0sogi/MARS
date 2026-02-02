import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# =============================================================================
# Mappings
# =============================================================================
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def get_structure_pairs(structure):
    """
    Parses dot-bracket structure to find pairs.
    Returns a mapping array where arr[i] = j if i is paired with j.
    If i is unpaired, arr[i] = i (map to self).
    """
    L = len(structure)
    pairs = np.arange(L)  # Default to self
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                pairs[i] = j
                pairs[j] = i
    return pairs


class RNADataset(Dataset):
    def __init__(self, inputs, partner_indices, targets=None, ids=None):
        """
        inputs: (N, Seq_Len, Channels) - Combined One-Hot + Partner Features
        partner_indices: (N, Seq_Len) - Indices for gathering
        targets: (N, Seq_Len, Num_Targets) - Ground truth (optional)
        ids: List of IDs (optional)
        """
        self.inputs = inputs
        self.partner_indices = partner_indices
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Convert to torch tensors
        # Inputs are float32
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)
        # Partner indices are long (integers)
        p_idx = torch.tensor(self.partner_indices[idx], dtype=torch.long)

        if self.targets is not None:
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            return x, p_idx, y
        else:
            # For test set, return ID as well if needed, but standard loader usually just returns data
            # We can return ID if needed for submission, but usually handled outside or via auxiliary list
            return x, p_idx


def one_hot_encode(seq, mapping, vocab_size):
    encoding = np.zeros((len(seq), vocab_size), dtype=np.float32)
    for i, char in enumerate(seq):
        if char in mapping:
            encoding[i, mapping[char]] = 1.0
    return encoding


def process_dataframe(df, is_test=False):
    """
    Processes a dataframe into numpy arrays for inputs, partner_indices, and targets.
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LENGTH

    # Calculate total input channels
    # Seq(4) + Struct(3) + Loop(7) + PartnerID(4)
    input_dim = Config.INPUT_CHANNELS

    # Initialize arrays
    inputs = np.zeros((num_samples, seq_len, input_dim), dtype=np.float32)
    partner_indices = np.zeros((num_samples, seq_len), dtype=np.int32)

    if not is_test:
        targets = np.zeros((num_samples, seq_len, Config.NUM_TARGETS), dtype=np.float32)
    else:
        targets = None

    ids = df["id"].values

    # Pre-compute column indices for feature concatenation
    # Layout: [Seq (4) | Struct (3) | Loop (7) | Partner (4)]
    idx_seq_start = 0
    idx_struct_start = 4
    idx_loop_start = 7
    idx_partner_start = 14

    for idx, row in df.iterrows():
        # 0. Basic info
        sequence = row["sequence"]
        structure = row["structure"]
        loop_type = row["predicted_loop_type"]

        # 1. Base Features
        seq_oh = one_hot_encode(sequence, SEQ_MAP, Config.VOCAB_SIZE_SEQ)
        struct_oh = one_hot_encode(structure, STRUCT_MAP, Config.VOCAB_SIZE_STRUCT)
        loop_oh = one_hot_encode(loop_type, LOOP_MAP, Config.VOCAB_SIZE_LOOP)

        # 2. Partner Features
        pairs = get_structure_pairs(structure)
        partner_indices[idx] = pairs

        # Partner Identity: One-hot of the paired base
        partner_oh = np.zeros((seq_len, Config.PARTNER_FEAT_SIZE), dtype=np.float32)
        for i, p_idx in enumerate(pairs):
            if i != p_idx:  # If paired
                partner_base = sequence[p_idx]
                if partner_base in SEQ_MAP:
                    partner_oh[i, SEQ_MAP[partner_base]] = 1.0

        # 3. Concatenate Inputs
        inputs[idx, :, idx_seq_start:idx_struct_start] = seq_oh
        inputs[idx, :, idx_struct_start:idx_loop_start] = struct_oh
        inputs[idx, :, idx_loop_start:idx_partner_start] = loop_oh
        inputs[idx, :, idx_partner_start:] = partner_oh

        # 4. Targets (Train/Val only)
        if not is_test:
            # Targets are provided as stringified lists for the first seq_scored positions
            # We need to parse them and pad to seq_len
            for t_i, col_name in enumerate(Config.ALL_TARGET_COLS):
                val_str = row[col_name]
                try:
                    val_list = ast.literal_eval(val_str)
                    # Assign to the first len(val_list) positions
                    # Usually len(val_list) == seq_scored (68)
                    length = len(val_list)
                    targets[idx, :length, t_i] = val_list
                except (ValueError, SyntaxError):
                    # Handle potential parsing errors or NaNs by leaving as 0
                    pass

    return inputs, partner_indices, targets, ids


def get_dataloaders(debug=False, load_cached_data=True):
    """
    Main function to load data, process/cache it, and return DataLoaders.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Define cache filenames
    cache_files = {
        "train": os.path.join(Config.CACHE_DIR, "train_data.npz"),
        "val": os.path.join(Config.CACHE_DIR, "val_data.npz"),
        "test": os.path.join(Config.CACHE_DIR, "test_data.npz"),
    }

    data = {}

    # -------------------------------------------------------------------------
    # Processing Logic
    # -------------------------------------------------------------------------
    splits = ["train", "val", "test"]

    for split in splits:
        cache_path = cache_files[split]
        is_test = split == "test"

        # Check if we can load from cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading {split} data from cache: {cache_path}")
            loaded = np.load(cache_path, allow_pickle=True)
            data[split] = {
                "inputs": loaded["inputs"],
                "partner_indices": loaded["partner_indices"],
                "ids": loaded["ids"],
            }
            if not is_test:
                data[split]["targets"] = loaded["targets"]
        else:
            print(f"Processing {split} data from source...")
            # Load source CSV
            if split == "train":
                csv_path = Config.TRAIN_DATA_PATH
            elif split == "val":
                csv_path = Config.VAL_DATA_PATH
            else:
                csv_path = Config.TEST_DATA_PATH

            df = pd.read_csv(csv_path)

            # Handle Debug Mode (Subsampling)
            if debug:
                df = df.head(Config.DEBUG_SUBSET_SIZE)

            # Process
            inputs, partner_indices, targets, ids = process_dataframe(
                df, is_test=is_test
            )

            # Save to Cache
            save_dict = {
                "inputs": inputs,
                "partner_indices": partner_indices,
                "ids": ids,
            }
            if not is_test:
                save_dict["targets"] = targets

            np.savez(cache_path, **save_dict)
            print(f"Saved {split} data to cache: {cache_path}")

            # Store in memory
            data[split] = save_dict

    # -------------------------------------------------------------------------
    # Create Datasets and Loaders
    # -------------------------------------------------------------------------

    # Train
    train_dataset = RNADataset(
        inputs=data["train"]["inputs"],
        partner_indices=data["train"]["partner_indices"],
        targets=data["train"]["targets"],
        ids=data["train"]["ids"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # Val
    val_dataset = RNADataset(
        inputs=data["val"]["inputs"],
        partner_indices=data["val"]["partner_indices"],
        targets=data["val"]["targets"],
        ids=data["val"]["ids"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Test
    test_dataset = RNADataset(
        inputs=data["test"]["inputs"],
        partner_indices=data["test"]["partner_indices"],
        targets=None,
        ids=data["test"]["ids"],
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
