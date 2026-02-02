import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import parse_list_column

# ==========================================
# Mappings for One-Hot Encoding
# ==========================================
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


class RNADataset(Dataset):
    def __init__(self, inputs, targets, partner_indices, masks, ids):
        """
        Args:
            inputs (np.ndarray): Shape (N, Seq_Len, 18). Static features.
            targets (np.ndarray): Shape (N, Seq_Len, 5). Ground truth.
            partner_indices (np.ndarray): Shape (N, Seq_Len). Indices of paired bases.
            masks (np.ndarray): Shape (N, Seq_Len). 1.0 for scored positions, 0.0 otherwise.
            ids (np.ndarray): Array of sample IDs.
        """
        self.inputs = inputs
        self.targets = targets
        self.partner_indices = partner_indices
        self.masks = masks
        self.ids = ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Convert to torch tensors
        # Inputs: (Seq_Len, 18)
        # Note: The model expects 23 channels. The training loop must append
        # the 5 recycling channels (initialized to 0 or from previous pass).
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)

        # Targets: (Seq_Len, 5)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)

        # Partner Indices: (Seq_Len,)
        # Used for gathering features from paired bases. -1 indicates unpaired.
        p_idx = torch.tensor(self.partner_indices[idx], dtype=torch.long)

        # Mask: (Seq_Len,)
        mask = torch.tensor(self.masks[idx], dtype=torch.float32)

        # ID
        sample_id = self.ids[idx]

        return x, y, p_idx, mask, sample_id


def process_data(df, mode="train"):
    """
    Processes the raw dataframe into numpy arrays suitable for training.
    """
    n_samples = len(df)
    seq_len = Config.SEQ_LENGTH

    # Static feature channels:
    # Sequence(4) + Structure(3) + Loop(7) + PartnerID(4) = 18
    n_channels = 18

    inputs = np.zeros((n_samples, seq_len, n_channels), dtype=np.float32)
    targets = np.zeros((n_samples, seq_len, 5), dtype=np.float32)
    partner_indices = np.full((n_samples, seq_len), -1, dtype=np.int32)
    masks = np.zeros((n_samples, seq_len), dtype=np.float32)
    ids = df["id"].values

    print(f"Processing {n_samples} samples for {mode}...")

    # Process each sample
    for i, row in df.iterrows():
        # 1. Sequence One-Hot (Channels 0-3)
        seq = row["sequence"]
        for j, char in enumerate(seq):
            if char in SEQ_MAP:
                inputs[i, j, SEQ_MAP[char]] = 1.0

        # 2. Structure One-Hot (Channels 4-6)
        struct = row["structure"]
        for j, char in enumerate(struct):
            if char in STRUCT_MAP:
                inputs[i, j, 4 + STRUCT_MAP[char]] = 1.0

        # 3. Loop Type One-Hot (Channels 7-13)
        loop = row["predicted_loop_type"]
        for j, char in enumerate(loop):
            if char in LOOP_MAP:
                inputs[i, j, 7 + LOOP_MAP[char]] = 1.0

        # 4. Partner Indices & Partner Identity (Channels 14-17)
        # Parse structure to find pairs
        stack = []
        pairs = {}
        for j, char in enumerate(struct):
            if char == "(":
                stack.append(j)
            elif char == ")":
                if stack:
                    k = stack.pop()
                    pairs[j] = k
                    pairs[k] = j

        for j in range(seq_len):
            if j in pairs:
                k = pairs[j]
                partner_indices[i, j] = k
                # Partner Identity: Copy the sequence one-hot of the partner
                # Sequence channels are 0-3, PartnerID channels are 14-17
                inputs[i, j, 14:18] = inputs[i, k, 0:4]
            else:
                partner_indices[i, j] = -1  # Unpaired
                # Partner Identity remains 0

        # 5. Targets & Masks
        if mode in ["train", "val"]:
            # Helper to safely get array from stringified list
            def get_target(col_name):
                val = row[col_name]
                return parse_list_column(val)

            t_react = get_target("reactivity")
            t_mg_ph10 = get_target("deg_Mg_pH10")
            t_ph10 = get_target("deg_pH10")
            t_mg_50c = get_target("deg_Mg_50C")
            t_50c = get_target("deg_50C")

            # Determine valid length (usually 68)
            scored_len = len(t_react)

            if scored_len > 0:
                # Fill targets
                targets[i, :scored_len, 0] = t_react
                targets[i, :scored_len, 1] = t_mg_ph10
                targets[i, :scored_len, 2] = t_ph10
                targets[i, :scored_len, 3] = t_mg_50c
                targets[i, :scored_len, 4] = t_50c

                # Mask
                masks[i, :scored_len] = 1.0
        else:
            # Test mode: Targets are 0, Mask based on seq_scored
            scored_len = row["seq_scored"]
            masks[i, :scored_len] = 1.0

    return inputs, targets, partner_indices, masks, ids


def get_loader(
    mode="train", batch_size=None, load_cached_data=True, shuffle=None, num_workers=None
):
    """
    Creates and returns a DataLoader for the requested mode.
    Handles caching of processed data to avoid re-computation.
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    if num_workers is None:
        num_workers = Config.NUM_WORKERS

    # Determine paths and defaults
    if mode == "train":
        csv_path = os.path.join(Config.METADATA_DIR, "train.csv")
        cache_path = Config.TRAIN_CACHE
        default_shuffle = True
    elif mode == "val":
        csv_path = os.path.join(Config.METADATA_DIR, "val.csv")
        cache_path = Config.VAL_CACHE
        default_shuffle = False
    elif mode == "test":
        csv_path = os.path.join(Config.METADATA_DIR, "test.csv")
        cache_path = Config.TEST_CACHE
        default_shuffle = False
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if shuffle is None:
        shuffle = default_shuffle

    # Ensure cache directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Try loading cache
    data_loaded = False
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading cached {mode} data from {cache_path}...")
            cached = np.load(cache_path, allow_pickle=True)
            inputs = cached["inputs"]
            targets = cached["targets"]
            partner_indices = cached["partner_indices"]
            masks = cached["masks"]
            ids = cached["ids"]
            data_loaded = True
            print("Cache loaded successfully.")
        except Exception as e:
            print(f"Failed to load cache: {e}. Re-processing...")

    if not data_loaded:
        print(f"Reading CSV from {csv_path}...")
        df = pd.read_csv(csv_path)

        # Debug subset logic
        if Config.DEBUG:
            print("DEBUG MODE: Using subset of 50 samples.")
            df = df.head(50)

        inputs, targets, partner_indices, masks, ids = process_data(df, mode=mode)

        # Save cache
        print(f"Saving cache to {cache_path}...")
        np.savez(
            cache_path,
            inputs=inputs,
            targets=targets,
            partner_indices=partner_indices,
            masks=masks,
            ids=ids,
        )

    dataset = RNADataset(inputs, targets, partner_indices, masks, ids)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader
