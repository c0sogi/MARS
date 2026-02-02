import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# =========================================================================
# Mappings
# =========================================================================
SEQ_MAP = {"A": 0, "G": 1, "U": 2, "C": 3}
STRUCT_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}
# Partner Identity: 0=A, 1=G, 2=U, 3=C, 4=None
PARTNER_ID_MAP = {"A": 0, "G": 1, "U": 2, "C": 3}


def get_partner_map_indices(structure):
    """
    Parses dot-bracket structure to find pair indices.
    Returns:
        np.ndarray: Array of shape (L,) where arr[i] is the index of the partner
                    of base i. If unpaired, arr[i] = -1.
    """
    L = len(structure)
    partner_map = np.full(L, -1, dtype=np.int32)
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


def one_hot_encode(seq, mapping, num_classes):
    """
    One-hot encodes a sequence string based on a mapping.
    Returns: (num_classes, L)
    """
    L = len(seq)
    arr = np.zeros((num_classes, L), dtype=np.float32)
    for i, char in enumerate(seq):
        if char in mapping:
            arr[mapping[char], i] = 1.0
    return arr


def process_data(df, mode="train"):
    """
    Process dataframe into numpy arrays for features and targets.
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LEN

    # Initialize containers
    # Features: (N, Channels, L)
    # Channels = Seq(4) + Struct(3) + Loop(7) + PartnerID(5) = 19
    all_inputs = np.zeros(
        (num_samples, Config.INPUT_CHANNELS, seq_len), dtype=np.float32
    )

    # Partner Map: (N, L) - indices for gathering
    all_partner_maps = np.zeros((num_samples, seq_len), dtype=np.int32)

    # Targets: (N, 5, L) - padded
    # We only have targets for train/val
    all_targets = None
    if mode in ["train", "val"]:
        all_targets = np.zeros(
            (num_samples, Config.NUM_TARGETS, seq_len), dtype=np.float32
        )

    sample_ids = []

    for idx, row in df.iterrows():
        sequence = row["sequence"]
        structure = row["structure"]
        loop_type = row["predicted_loop_type"]

        # 1. Basic One-Hot Features
        oh_seq = one_hot_encode(sequence, SEQ_MAP, Config.NUM_SEQUENCE_TYPES)
        oh_struct = one_hot_encode(structure, STRUCT_MAP, Config.NUM_STRUCTURE_TYPES)
        oh_loop = one_hot_encode(loop_type, LOOP_MAP, Config.NUM_LOOP_TYPES)

        # 2. Partner Map & Partner Identity
        p_map = get_partner_map_indices(structure)

        # Generate Partner Identity Feature (5 channels)
        oh_partner = np.zeros((Config.NUM_PARTNER_TYPES, seq_len), dtype=np.float32)

        # Prepare gather map: replace -1 with i (self) to avoid gather errors
        # The model should mask these positions based on structure input or logic
        gather_indices = np.copy(p_map)

        for i in range(seq_len):
            partner_idx = p_map[i]
            if partner_idx != -1:
                # Paired: get partner's base identity
                partner_base = sequence[partner_idx]
                if partner_base in PARTNER_ID_MAP:
                    oh_partner[PARTNER_ID_MAP[partner_base], i] = 1.0
            else:
                # Unpaired: set last channel (None)
                oh_partner[4, i] = 1.0
                # Fix gather index to self for safety
                gather_indices[i] = i

        # Concatenate all features
        # Shape: (19, 107)
        combined_features = np.concatenate(
            [oh_seq, oh_struct, oh_loop, oh_partner], axis=0
        )

        all_inputs[idx] = combined_features
        all_partner_maps[idx] = gather_indices
        sample_ids.append(row["id"])

        # 3. Targets (if available)
        if mode in ["train", "val"]:
            # Targets are provided as strings of lists in the CSV
            # We need to parse them and pad to seq_len
            for t_i, col in enumerate(Config.TARGET_COLS):
                val_str = row[col]
                try:
                    val_list = ast.literal_eval(val_str)
                    # Copy into the target array (first 68 positions)
                    length = min(len(val_list), seq_len)
                    all_targets[idx, t_i, :length] = val_list
                except:
                    pass  # Keep as zeros if error or empty

    return {
        "inputs": all_inputs,
        "partner_maps": all_partner_maps,
        "targets": all_targets,
        "ids": np.array(sample_ids),
    }


class RNADataset(Dataset):
    def __init__(self, data_dict, mode="train"):
        self.inputs = data_dict["inputs"]
        self.partner_maps = data_dict["partner_maps"]
        self.ids = data_dict["ids"]
        self.mode = mode

        if mode in ["train", "val"]:
            self.targets = data_dict["targets"]
        else:
            self.targets = None

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Inputs: (C, L)
        x = torch.from_numpy(self.inputs[idx])

        # Partner Map: (L,)
        p_map = torch.from_numpy(self.partner_maps[idx]).long()

        if self.mode in ["train", "val"]:
            # Targets: (5, L)
            y = torch.from_numpy(self.targets[idx])
            return x, p_map, y
        else:
            sample_id = self.ids[idx]
            return x, p_map, sample_id


def get_loaders(load_cached_data=True):
    """
    Main function to load data, process/cache it, and return DataLoaders.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # ---------------------------------------------------------
    # 1. TRAIN DATA
    # ---------------------------------------------------------
    if load_cached_data and os.path.exists(Config.TRAIN_CACHE):
        print(f"Loading cached train data from {Config.TRAIN_CACHE}")
        train_data = np.load(Config.TRAIN_CACHE, allow_pickle=True)
        train_dict = {k: train_data[k] for k in train_data.files}
    else:
        print("Processing train data...")
        df_train = pd.read_csv(os.path.join(Config.METADATA_DIR, "train.csv"))
        train_dict = process_data(df_train, mode="train")
        np.savez(Config.TRAIN_CACHE, **train_dict)
        print(f"Saved train cache to {Config.TRAIN_CACHE}")

    # ---------------------------------------------------------
    # 2. VAL DATA
    # ---------------------------------------------------------
    if load_cached_data and os.path.exists(Config.VAL_CACHE):
        print(f"Loading cached val data from {Config.VAL_CACHE}")
        val_data = np.load(Config.VAL_CACHE, allow_pickle=True)
        val_dict = {k: val_data[k] for k in val_data.files}
    else:
        print("Processing val data...")
        df_val = pd.read_csv(os.path.join(Config.METADATA_DIR, "val.csv"))
        val_dict = process_data(df_val, mode="val")
        np.savez(Config.VAL_CACHE, **val_dict)
        print(f"Saved val cache to {Config.VAL_CACHE}")

    # ---------------------------------------------------------
    # 3. TEST DATA
    # ---------------------------------------------------------
    if load_cached_data and os.path.exists(Config.TEST_CACHE):
        print(f"Loading cached test data from {Config.TEST_CACHE}")
        test_data = np.load(Config.TEST_CACHE, allow_pickle=True)
        test_dict = {k: test_data[k] for k in test_data.files}
    else:
        print("Processing test data...")
        df_test = pd.read_csv(os.path.join(Config.METADATA_DIR, "test.csv"))
        test_dict = process_data(df_test, mode="test")
        np.savez(Config.TEST_CACHE, **test_dict)
        print(f"Saved test cache to {Config.TEST_CACHE}")

    # ---------------------------------------------------------
    # 4. Create Datasets and Loaders
    # ---------------------------------------------------------
    train_dataset = RNADataset(train_dict, mode="train")
    val_dataset = RNADataset(val_dict, mode="val")
    test_dataset = RNADataset(test_dict, mode="test")

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
