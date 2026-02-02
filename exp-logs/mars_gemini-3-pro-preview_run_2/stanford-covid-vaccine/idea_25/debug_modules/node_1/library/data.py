import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import parse_list_column

# --------------------------------------------------------------------------
# Mappings
# --------------------------------------------------------------------------
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


# --------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------
def get_structure_pairs(structure):
    """
    Parses a dot-bracket structure string to find base pairs.
    Returns an array of partner indices. Unpaired bases are -1.
    """
    pairs = np.full(len(structure), -1, dtype=np.int64)
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


def one_hot(seq, map_dict, num_classes):
    """
    One-hot encodes a sequence string based on a mapping dictionary.
    """
    encoding = np.zeros((len(seq), num_classes), dtype=np.float32)
    for i, char in enumerate(seq):
        if char in map_dict:
            encoding[i, map_dict[char]] = 1.0
    return encoding


# --------------------------------------------------------------------------
# Dataset Class
# --------------------------------------------------------------------------
class RNADataset(Dataset):
    def __init__(self, inputs, partner_indices, targets=None, masks=None, ids=None):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        # Partner indices: -1 indicates no partner.
        # Model handles masking or clamping based on this.
        self.partner_indices = torch.tensor(partner_indices, dtype=torch.long)

        if targets is not None:
            self.targets = torch.tensor(targets, dtype=torch.float32)
        else:
            self.targets = None

        if masks is not None:
            self.masks = torch.tensor(masks, dtype=torch.float32)
        else:
            self.masks = None

        self.ids = ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        sample = {
            "inputs": self.inputs[idx],
            "partner_indices": self.partner_indices[idx],
        }

        if self.targets is not None:
            sample["targets"] = self.targets[idx]

        if self.masks is not None:
            sample["mask"] = self.masks[idx]

        if self.ids is not None:
            sample["id"] = self.ids[idx]

        return sample


# --------------------------------------------------------------------------
# Data Preprocessing
# --------------------------------------------------------------------------
def preprocess_data(csv_path, cache_path, load_cached_data=True, is_test=False):
    """
    Loads data from CSV, generates features, and caches them to an NPZ file.

    Args:
        csv_path (str): Path to the source CSV file.
        cache_path (str): Path to save/load the processed .npz file.
        load_cached_data (bool): Whether to attempt loading from cache.
        is_test (bool): Whether processing the test set (no targets).

    Returns:
        tuple: (inputs, partner_indices, targets, masks, ids)
    """
    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=True)
            inputs = data["inputs"]
            partner_indices = data["partner_indices"]
            ids = data["ids"]

            if not is_test:
                # Cite debug_lesson_9: Enforce schema validation. Required keys must exist.
                targets = data["targets"]
                masks = data["masks"]
            else:
                targets = None
                masks = None

            print(f"Loaded data from cache: {cache_path}")
            return inputs, partner_indices, targets, masks, ids
        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}. Recomputing...")

    # 2. Load CSV
    print(f"Processing data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # 3. Initialize Arrays
    num_samples = len(df)
    seq_len = Config.SEQ_LENGTH

    # Feature Dimensions: Seq(4) + Struct(3) + Loop(7) + PartnerID(5) = 19
    input_dim = Config.INPUT_DIM

    inputs = np.zeros((num_samples, seq_len, input_dim), dtype=np.float32)
    partner_indices_arr = np.zeros((num_samples, seq_len), dtype=np.int64)
    ids = df["id"].values

    targets = None
    masks = None

    if not is_test:
        targets = np.zeros((num_samples, seq_len, Config.NUM_TARGETS), dtype=np.float32)
        masks = np.zeros((num_samples, seq_len), dtype=np.float32)

    # 4. Iterate and Process
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for idx, row in df.iterrows():
        sequence = row["sequence"]
        structure = row["structure"]
        loop_type = row["predicted_loop_type"]

        # --- Feature Engineering ---

        # A. Basic One-Hot Encodings
        ohe_seq = one_hot(sequence, SEQ_MAP, 4)
        ohe_struct = one_hot(structure, STRUCT_MAP, 3)
        ohe_loop = one_hot(loop_type, LOOP_MAP, 7)

        # B. Partner Indices
        pairs = get_structure_pairs(structure)
        partner_indices_arr[idx] = pairs

        # C. Partner Identity (5)
        # 0-3: A, G, C, U
        # 4: No Partner
        ohe_partner = np.zeros((seq_len, 5), dtype=np.float32)

        for i, p_idx in enumerate(pairs):
            if p_idx != -1:
                # Get the base at the partner index
                partner_base = sequence[p_idx]
                if partner_base in SEQ_MAP:
                    ohe_partner[i, SEQ_MAP[partner_base]] = 1.0
            else:
                # No partner
                ohe_partner[i, 4] = 1.0

        # Concatenate all features
        # Shape: (Seq_Len, 19)
        sample_input = np.concatenate(
            [ohe_seq, ohe_struct, ohe_loop, ohe_partner], axis=1
        )
        inputs[idx] = sample_input

        # --- Targets & Masks ---
        if not is_test:
            # Scoring mask: 1 for scored positions, 0 otherwise
            scored_len = int(row["seq_scored"])
            sample_mask = np.zeros(seq_len, dtype=np.float32)
            sample_mask[:scored_len] = 1.0
            masks[idx] = sample_mask

            # Initialize sample targets
            sample_targets = np.zeros((seq_len, Config.NUM_TARGETS), dtype=np.float32)

            for t_i, col in enumerate(target_cols):
                val_arr = parse_list_column(row[col])
                # Ensure length matches scored_len and fits in sequence
                length = min(len(val_arr), seq_len)
                sample_targets[:length, t_i] = val_arr[:length]

            targets[idx] = sample_targets

    # 5. Save to Cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    save_dict = {"inputs": inputs, "partner_indices": partner_indices_arr, "ids": ids}
    if targets is not None:
        save_dict["targets"] = targets
    if masks is not None:
        save_dict["masks"] = masks

    np.savez_compressed(cache_path, **save_dict)
    print(f"Saved processed data to {cache_path}")

    return inputs, partner_indices_arr, targets, masks, ids


def get_dataloaders(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, debug=Config.DEBUG
):
    """
    Prepares DataLoaders for train, validation, and test sets.
    """

    # Train
    train_inputs, train_partners, train_targets, train_masks, train_ids = (
        preprocess_data(
            Config.TRAIN_CSV, Config.TRAIN_CACHE, load_cached_data=True, is_test=False
        )
    )

    # Val
    val_inputs, val_partners, val_targets, val_masks, val_ids = preprocess_data(
        Config.VAL_CSV, Config.VAL_CACHE, load_cached_data=True, is_test=False
    )

    # Test
    test_inputs, test_partners, _, _, test_ids = preprocess_data(
        Config.TEST_CSV, Config.TEST_CACHE, load_cached_data=True, is_test=True
    )

    # Debugging: Reduce dataset size
    if debug:
        limit = Config.DEBUG_SAMPLES
        train_inputs = train_inputs[:limit]
        train_partners = train_partners[:limit]
        train_targets = train_targets[:limit]
        train_masks = train_masks[:limit]
        train_ids = train_ids[:limit]

        val_inputs = val_inputs[:limit]
        val_partners = val_partners[:limit]
        val_targets = val_targets[:limit]
        val_masks = val_masks[:limit]
        val_ids = val_ids[:limit]

    # Create Datasets
    train_dataset = RNADataset(
        train_inputs, train_partners, train_targets, train_masks, train_ids
    )
    val_dataset = RNADataset(val_inputs, val_partners, val_targets, val_masks, val_ids)
    test_dataset = RNADataset(test_inputs, test_partners, ids=test_ids)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
