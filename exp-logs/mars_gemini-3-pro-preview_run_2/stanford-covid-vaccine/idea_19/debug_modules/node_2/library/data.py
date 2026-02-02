import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# ==========================================
# Constants & Mappings
# ==========================================
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

# Inverse sequence map for looking up partner identity
IDX_TO_BASE = {v: k for k, v in SEQ_MAP.items()}


# ==========================================
# Helper Functions
# ==========================================
def parse_structure_pairs(structure_str):
    """
    Parses a dot-bracket structure string to find base pairs.
    Returns a numpy array of shape (len(structure),) where arr[i] is the index
    of the base paired with i, or -1 if unpaired.
    """
    n = len(structure_str)
    partner_indices = np.full(n, -1, dtype=np.int32)
    stack = []

    for i, char in enumerate(structure_str):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                partner_indices[i] = j
                partner_indices[j] = i
            else:
                # Unbalanced closing parenthesis, technically shouldn't happen in valid data
                pass

    return partner_indices


def one_hot_encode(seq, mapping, length):
    """
    One-hot encodes a sequence string based on a mapping.
    Returns (length, num_classes).
    """
    num_classes = len(mapping)
    encoding = np.zeros((length, num_classes), dtype=np.float32)
    for i, char in enumerate(seq):
        if i >= length:
            break
        if char in mapping:
            encoding[i, mapping[char]] = 1.0
    return encoding


def get_partner_identity_features(sequence, partner_indices):
    """
    Generates one-hot encoding of the paired base.
    If unpaired (index -1), returns zero vector.
    """
    length = len(sequence)
    num_classes = len(SEQ_MAP)
    features = np.zeros((length, num_classes), dtype=np.float32)

    for i, partner_idx in enumerate(partner_indices):
        if partner_idx != -1:
            partner_base = sequence[partner_idx]
            if partner_base in SEQ_MAP:
                features[i, SEQ_MAP[partner_base]] = 1.0

    return features


def parse_target_list(target_str, length=None):
    """
    Parses a string representation of a list into a numpy array.
    Pads with zeros to `length` if provided.
    """
    try:
        vals = ast.literal_eval(target_str)
        arr = np.array(vals, dtype=np.float32)
    except (ValueError, SyntaxError):
        arr = np.array([], dtype=np.float32)

    if length is not None:
        if len(arr) < length:
            pad_width = length - len(arr)
            arr = np.pad(arr, (0, pad_width), mode="constant", constant_values=0)
        elif len(arr) > length:
            arr = arr[:length]

    return arr


# ==========================================
# Data Processing & Caching
# ==========================================
def process_dataframe(df, mode="train"):
    """
    Processes a dataframe into numpy arrays for inputs, partner maps, and targets.
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LENGTH

    # Input feature dimensions:
    # Seq(4) + Struct(3) + Loop(7) + PartnerID(4) = 18 channels
    input_dim = 4 + 3 + 7 + 4

    inputs = np.zeros((num_samples, seq_len, input_dim), dtype=np.float32)
    partner_maps = np.zeros((num_samples, seq_len), dtype=np.int32)

    # Targets: 5 columns
    if mode in ["train", "val"]:
        targets = np.zeros((num_samples, seq_len, 5), dtype=np.float32)
    else:
        targets = None

    ids = df["id"].values

    for idx, row in df.iterrows():
        # 1. Basic Sequence/Structure Parsing
        sequence = row["sequence"]
        structure = row["structure"]
        loop_type = row["predicted_loop_type"]

        # 2. Generate One-Hot Encodings
        enc_seq = one_hot_encode(sequence, SEQ_MAP, seq_len)
        enc_struct = one_hot_encode(structure, STRUCT_MAP, seq_len)
        enc_loop = one_hot_encode(loop_type, LOOP_MAP, seq_len)

        # 3. Partner Logic
        p_indices = parse_structure_pairs(structure)
        enc_partner = get_partner_identity_features(sequence, p_indices)

        # 4. Concatenate Inputs
        # Shape: (SeqLen, 18)
        sample_input = np.concatenate(
            [enc_seq, enc_struct, enc_loop, enc_partner], axis=1
        )

        # Store Inputs
        # Row index in dataframe might not be 0..N if it's a slice, so use enumeration or reset index
        # We'll use a separate counter or assume df is 0..N index reset.
        # Safer to use the loop index if we iterate via range, but iterrows is used.
        # Let's use the integer location `idx` if the df index is reset,
        # but to be safe, we'll use a counter.

        # Actually, let's just iterate with enumerate on values to be fast and safe
        pass

    # Re-implementing loop for speed and safety
    for i in range(num_samples):
        row = df.iloc[i]
        sequence = row["sequence"]
        structure = row["structure"]
        loop_type = row["predicted_loop_type"]

        # Encodings
        enc_seq = one_hot_encode(sequence, SEQ_MAP, seq_len)
        enc_struct = one_hot_encode(structure, STRUCT_MAP, seq_len)
        enc_loop = one_hot_encode(loop_type, LOOP_MAP, seq_len)

        # Partner Info
        p_indices = parse_structure_pairs(structure)
        enc_partner = get_partner_identity_features(sequence, p_indices)

        # Combine
        inputs[i] = np.concatenate([enc_seq, enc_struct, enc_loop, enc_partner], axis=1)
        partner_maps[i] = p_indices

        # Targets
        if mode in ["train", "val"]:
            # Columns: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
            # Note: The order must match Config.ALL_TARGETS
            t_list = []
            for col in Config.ALL_TARGETS:
                val_arr = parse_target_list(row[col], length=seq_len)
                t_list.append(val_arr)

            # Stack along last axis -> (SeqLen, 5)
            targets[i] = np.stack(t_list, axis=1)

    return inputs, partner_maps, targets, ids


def load_or_process_data(split_name, df, load_cached_data=True):
    """
    Handles caching logic.
    """
    cache_dir = Config.WORKING_DIR
    cache_key = f"{split_name}_data_{Config.CACHE_VERSION}.npz"
    cache_path = os.path.join(cache_dir, cache_key)

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {split_name} data from {cache_path}...")
        try:
            data = np.load(cache_path, allow_pickle=True)
            inputs = data["inputs"]
            partner_maps = data["partner_maps"]
            targets = data["targets"] if "targets" in data else None
            ids = data["ids"]
            return inputs, partner_maps, targets, ids
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    print(f"Processing {split_name} data...")
    inputs, partner_maps, targets, ids = process_dataframe(df, mode=split_name)

    print(f"Saving {split_name} data to {cache_path}...")
    save_dict = {"inputs": inputs, "partner_maps": partner_maps, "ids": ids}
    if targets is not None:
        save_dict["targets"] = targets

    np.savez_compressed(cache_path, **save_dict)

    return inputs, partner_maps, targets, ids


# ==========================================
# Dataset Class
# ==========================================
class RNADataset(Dataset):
    def __init__(self, inputs, partner_maps, targets=None, ids=None):
        self.inputs = inputs
        self.partner_maps = partner_maps
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Inputs: (SeqLen, Channels)
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)

        # Partner Map: (SeqLen,)
        # Note: We use LongTensor for indices
        p_map = torch.tensor(self.partner_maps[idx], dtype=torch.long)

        # Targets
        if self.targets is not None:
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            return x, p_map, y
        else:
            # For inference, just return inputs and partner map
            return x, p_map


# ==========================================
# DataLoader Factory
# ==========================================
def get_dataloaders(debug=False, load_cached_data=True):
    """
    Main entry point to get DataLoaders.
    """
    # Load Metadata
    train_csv = os.path.join(Config.METADATA_DIR, "train.csv")
    val_csv = os.path.join(Config.METADATA_DIR, "val.csv")
    test_csv = os.path.join(Config.METADATA_DIR, "test.csv")

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_test = pd.read_csv(test_csv)

    # Debug Mode: Subset data
    if debug:
        df_train = df_train.iloc[:100]
        df_val = df_val.iloc[:20]
        df_test = df_test.iloc[:20]

    # Process Data
    train_inputs, train_pmaps, train_targets, _ = load_or_process_data(
        "train", df_train, load_cached_data
    )
    val_inputs, val_pmaps, val_targets, _ = load_or_process_data(
        "val", df_val, load_cached_data
    )
    test_inputs, test_pmaps, _, test_ids = load_or_process_data(
        "test", df_test, load_cached_data
    )

    # Create Datasets
    train_dataset = RNADataset(train_inputs, train_pmaps, train_targets)
    val_dataset = RNADataset(val_inputs, val_pmaps, val_targets)
    # For test dataset, we might want to access IDs later, but DataLoader usually just yields tensors.
    # We can handle ID mapping during inference by iterating the dataframe or passing IDs if needed.
    # Here we stick to the standard tensor interface.
    test_dataset = RNADataset(test_inputs, test_pmaps, targets=None, ids=test_ids)

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
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
