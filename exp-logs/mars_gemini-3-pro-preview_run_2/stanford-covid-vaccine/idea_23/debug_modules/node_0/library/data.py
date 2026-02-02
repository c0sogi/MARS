import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# ------------------------------------------------------------------------------
# Mappings
# ------------------------------------------------------------------------------
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {".": 0, "(": 1, ")": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}
# Partner Identity: 0=A, 1=G, 2=C, 3=U, 4=None
PARTNER_BASE_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA sequences.
    Returns:
        inputs (torch.Tensor): (Seq_Len, Num_Features)
        partner_indices (torch.Tensor): (Seq_Len,) - Indices of paired bases
        targets (torch.Tensor): (Seq_Len, Num_Targets)
        ids (str): Sample ID
    """

    def __init__(self, inputs, partner_indices, targets, ids):
        self.inputs = inputs
        self.partner_indices = partner_indices
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Convert to tensors
        # Inputs: Float32
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)

        # Partner Indices: Long
        # We clamp -1 to 0 to ensure valid indices for gather operations.
        # The model is expected to use the structure/partner-identity features
        # to mask out these dummy connections.
        p_idx_raw = self.partner_indices[idx]
        p_idx = torch.tensor(np.maximum(p_idx_raw, 0), dtype=torch.long)

        # Targets: Float32
        y = torch.tensor(self.targets[idx], dtype=torch.float32)

        # ID: String
        sample_id = self.ids[idx]

        return x, p_idx, y, sample_id


def get_couples(structure):
    """
    Parses a dot-bracket structure string to find base pairs.
    Returns a dictionary mapping index -> partner_index.
    """
    pairs = {}
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


def process_dataframe(df, is_test=False):
    """
    Processes a dataframe into numpy arrays for inputs, partner_indices, and targets.
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LENGTH

    # Initialize arrays
    # Features: Seq(4) + Struct(3) + Loop(7) + PartnerID(5) = 19
    num_features = Config.NUM_NODE_FEATURES

    inputs = np.zeros((num_samples, seq_len, num_features), dtype=np.float32)
    partner_indices = np.full((num_samples, seq_len), -1, dtype=np.int32)
    targets = np.zeros((num_samples, seq_len, Config.NUM_TARGETS), dtype=np.float32)
    ids = df["id"].values

    # Pre-compute one-hot identity matrices for speed
    # Sequence: 4 -> 4
    eye_seq = np.eye(4, dtype=np.float32)
    # Structure: 3 -> 3
    eye_struct = np.eye(3, dtype=np.float32)
    # Loop: 7 -> 7
    eye_loop = np.eye(7, dtype=np.float32)
    # PartnerID: 5 -> 5
    eye_partner = np.eye(5, dtype=np.float32)

    for idx, row in df.iterrows():
        # 1. Parse Basic Sequences
        seq_str = row["sequence"]
        struct_str = row["structure"]
        loop_str = row["predicted_loop_type"]

        # 2. Parse Pairs
        pairs_map = get_couples(struct_str)

        # 3. Build Features per Position
        for i in range(seq_len):
            # A. Sequence One-Hot
            base = seq_str[i]
            if base in SEQ_MAP:
                inputs[idx, i, 0:4] = eye_seq[SEQ_MAP[base]]

            # B. Structure One-Hot
            struct_char = struct_str[i]
            if struct_char in STRUCT_MAP:
                inputs[idx, i, 4:7] = eye_struct[STRUCT_MAP[struct_char]]

            # C. Loop Type One-Hot
            loop_char = loop_str[i]
            if loop_char in LOOP_MAP:
                inputs[idx, i, 7:14] = eye_loop[LOOP_MAP[loop_char]]

            # D. Partner Identity & Index
            if i in pairs_map:
                partner_idx = pairs_map[i]
                partner_indices[idx, i] = partner_idx

                partner_base = seq_str[partner_idx]
                if partner_base in PARTNER_BASE_MAP:
                    # Map A,G,C,U to 0-3
                    inputs[idx, i, 14:19] = eye_partner[PARTNER_BASE_MAP[partner_base]]
                else:
                    # Fallback (shouldn't happen for valid RNA)
                    inputs[idx, i, 14:19] = eye_partner[4]
            else:
                # Unpaired: Partner Identity is "None" (index 4)
                inputs[idx, i, 14:19] = eye_partner[4]
                # Partner Index remains -1

        # 4. Parse Targets (if not test)
        if not is_test:
            # Targets are stored as stringified lists in CSV
            # We need to parse them.
            # Columns: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
            for t_i, col in enumerate(Config.TARGET_COLS):
                val_str = row[col]
                try:
                    val_list = ast.literal_eval(val_str)
                    # The provided values are for the first seq_scored positions
                    length = len(val_list)
                    targets[idx, :length, t_i] = val_list
                except (ValueError, SyntaxError):
                    # Handle cases where parsing fails or data is missing
                    pass

    return inputs, partner_indices, targets, ids


def get_data(debug=False, load_cached_data=True):
    """
    Main function to load and process data.
    Manages caching to avoid re-processing.
    """
    # Define Cache Paths
    train_cache = Config.TRAIN_CACHE
    val_cache = Config.VAL_CACHE
    test_cache = Config.TEST_CACHE

    # Helper to load or process
    def load_or_process(csv_path, cache_path, is_test=False, subset=None):
        # Try loading cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached data from {cache_path}...")
            try:
                data = np.load(cache_path, allow_pickle=True)
                return (
                    data["inputs"],
                    data["partner_indices"],
                    data["targets"],
                    data["ids"],
                )
            except Exception as e:
                print(f"Failed to load cache: {e}. Re-processing...")

        # Process from scratch
        print(f"Processing data from {csv_path}...")
        df = pd.read_csv(csv_path)

        if subset:
            df = df.iloc[:subset]

        inputs, partner_indices, targets, ids = process_dataframe(df, is_test=is_test)

        # Save cache
        print(f"Saving cache to {cache_path}...")
        np.savez_compressed(
            cache_path,
            inputs=inputs,
            partner_indices=partner_indices,
            targets=targets,
            ids=ids,
        )

        return inputs, partner_indices, targets, ids

    # Determine subset size for debugging
    subset_size = Config.SUBSET_SIZE if debug else None

    # Load Train
    train_inputs, train_pidx, train_targets, train_ids = load_or_process(
        Config.TRAIN_CSV, train_cache, is_test=False, subset=subset_size
    )

    # Load Val
    val_inputs, val_pidx, val_targets, val_ids = load_or_process(
        Config.VAL_CSV, val_cache, is_test=False, subset=subset_size
    )

    # Load Test
    test_inputs, test_pidx, test_targets, test_ids = load_or_process(
        Config.TEST_CSV, test_cache, is_test=True, subset=subset_size
    )

    return (
        (train_inputs, train_pidx, train_targets, train_ids),
        (val_inputs, val_pidx, val_targets, val_ids),
        (test_inputs, test_pidx, test_targets, test_ids),
    )


def get_dataloaders(debug=False, load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.
    """
    (train_data, val_data, test_data) = get_data(debug, load_cached_data)

    # Create Datasets
    train_dataset = RNADataset(*train_data)
    val_dataset = RNADataset(*val_data)
    test_dataset = RNADataset(*test_data)

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader
