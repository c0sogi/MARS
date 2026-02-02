import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class RNADataset(Dataset):
    def __init__(
        self,
        inputs,
        targets,
        masks,
        partner_indices,
        pairing_masks,
        ids,
        mode="train",
    ):
        self.inputs = inputs
        self.targets = targets
        self.masks = masks
        self.partner_indices = partner_indices
        self.pairing_masks = pairing_masks
        self.ids = ids
        self.mode = mode

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Convert to torch tensors
        # inputs: (Seq_Len, 18)
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)

        # targets: (Seq_Len, 5)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)

        # mask: (Seq_Len,) - 1 for scored positions, 0 otherwise
        mask = torch.tensor(self.masks[idx], dtype=torch.float32)

        # partner_indices: (Seq_Len,) - Index of partner, or self if unpaired
        p_idx = torch.tensor(self.partner_indices[idx], dtype=torch.long)

        # pairing_mask: (Seq_Len,) - 1 if paired, 0 if unpaired
        p_mask = torch.tensor(self.pairing_masks[idx], dtype=torch.float32)

        sample_id = self.ids[idx]

        return x, y, mask, p_idx, p_mask, sample_id


def get_couples(structure):
    """
    Parses a dot-bracket structure string to identify base pairs.
    Returns a dictionary mapping index i -> partner index j.
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


def preprocess_data(df, mode="train"):
    """
    Generates features and targets from the dataframe.

    Features (18 channels):
    - Sequence (4): A, G, C, U
    - Structure (3): (, ), .
    - Loop Type (7): S, M, I, B, H, E, X
    - Partner Identity (4): A, G, C, U of the paired base (0 if unpaired)

    Auxiliary:
    - Partner Indices: Index of paired base (or self if unpaired)
    - Pairing Mask: 1 if paired, 0 if unpaired
    """
    # Mappings
    seq_map = {"A": 0, "G": 1, "C": 2, "U": 3}
    struct_map = {"(": 0, ")": 1, ".": 2}
    loop_map = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

    num_samples = len(df)
    seq_len = Config.SEQ_LENGTH

    # Initialize arrays
    # Inputs: (N, L, 18)
    inputs = np.zeros((num_samples, seq_len, 18), dtype=np.float32)
    # Targets: (N, L, 5)
    targets = np.zeros((num_samples, seq_len, 5), dtype=np.float32)
    # Masks: (N, L)
    masks = np.zeros((num_samples, seq_len), dtype=np.float32)
    # Partner Indices: (N, L)
    partner_indices = np.zeros((num_samples, seq_len), dtype=np.int32)
    # Pairing Masks: (N, L)
    pairing_masks = np.zeros((num_samples, seq_len), dtype=np.float32)

    ids = df["id"].values

    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for idx, row in df.iterrows():
        sequence = row["sequence"]
        structure = row["structure"]
        loop_type = row["predicted_loop_type"]
        seq_scored = row["seq_scored"]

        # 1. Base Features
        for i in range(seq_len):
            # Sequence One-Hot (0-3)
            if i < len(sequence):
                char = sequence[i]
                if char in seq_map:
                    inputs[idx, i, seq_map[char]] = 1.0

            # Structure One-Hot (4-6)
            if i < len(structure):
                char = structure[i]
                if char in struct_map:
                    inputs[idx, i, 4 + struct_map[char]] = 1.0

            # Loop Type One-Hot (7-13)
            if i < len(loop_type):
                char = loop_type[i]
                if char in loop_map:
                    inputs[idx, i, 7 + loop_map[char]] = 1.0

        # 2. Partner Identity & Indices
        pairs = get_couples(structure)
        for i in range(seq_len):
            if i in pairs:
                j = pairs[i]
                partner_indices[idx, i] = j
                pairing_masks[idx, i] = 1.0

                # Partner Identity One-Hot (14-17)
                if j < len(sequence):
                    partner_char = sequence[j]
                    if partner_char in seq_map:
                        inputs[idx, i, 14 + seq_map[partner_char]] = 1.0
            else:
                # If unpaired, point to self (to avoid gather errors) and mask out later
                partner_indices[idx, i] = i
                pairing_masks[idx, i] = 0.0
                # Partner Identity remains 0

        # 3. Targets & Mask
        if mode in ["train", "val"]:
            # Mask first 'seq_scored' positions
            masks[idx, :seq_scored] = 1.0

            for t_idx, col in enumerate(target_cols):
                # Parse stringified list
                val_str = row[col]
                try:
                    val_list = ast.literal_eval(val_str)
                    # Assign to first seq_scored positions
                    length = min(len(val_list), seq_len)
                    targets[idx, :length, t_idx] = np.array(val_list, dtype=np.float32)[
                        :length
                    ]
                except (ValueError, SyntaxError):
                    # Handle potential parsing errors or NaNs by leaving as 0
                    pass
        else:
            # Test mode: No targets, but mask determines what we submit (though we predict all)
            # We usually predict all 107, but the metric is only on scored.
            # For submission, we just output everything.
            masks[idx, :] = 1.0  # Technically not used for loss in test

    return inputs, targets, masks, partner_indices, pairing_masks, ids


def load_data(mode="train", load_cached_data=True):
    """
    Loads data from cache or processes from raw CSVs.
    """
    # Determine paths
    if mode == "train":
        csv_path = Config.TRAIN_CSV
        cache_path = Config.TRAIN_CACHE
    elif mode == "val":
        csv_path = Config.VAL_CSV
        cache_path = Config.VAL_CACHE
    elif mode == "test":
        csv_path = Config.TEST_CSV
        cache_path = Config.TEST_CACHE
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {mode} data from {cache_path}...")
        try:
            data = np.load(cache_path, allow_pickle=True)
            return RNADataset(
                inputs=data["inputs"],
                targets=data["targets"],
                masks=data["masks"],
                partner_indices=data["partner_indices"],
                pairing_masks=data["pairing_masks"],
                ids=data["ids"],
                mode=mode,
            )
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from Scratch
    print(f"Processing {mode} data from {csv_path}...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Preprocess
    inputs, targets, masks, partner_indices, pairing_masks, ids = preprocess_data(
        df, mode=mode
    )

    # 3. Save to Cache
    print(f"Saving {mode} data to {cache_path}...")
    np.savez_compressed(
        cache_path,
        inputs=inputs,
        targets=targets,
        masks=masks,
        partner_indices=partner_indices,
        pairing_masks=pairing_masks,
        ids=ids,
    )

    return RNADataset(
        inputs=inputs,
        targets=targets,
        masks=masks,
        partner_indices=partner_indices,
        pairing_masks=pairing_masks,
        ids=ids,
        mode=mode,
    )


def get_dataloaders(
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
):
    """
    Returns DataLoaders for train, val, and test sets.
    """
    # Load Datasets
    train_dataset = load_data("train", load_cached_data)
    val_dataset = load_data("val", load_cached_data)
    test_dataset = load_data("test", load_cached_data)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader
