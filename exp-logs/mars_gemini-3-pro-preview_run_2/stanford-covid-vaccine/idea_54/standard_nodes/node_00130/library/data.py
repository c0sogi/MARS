import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# Dictionaries for One-Hot Encoding
TOKEN_DICT = {
    "sequence": {"A": 0, "G": 1, "C": 2, "U": 3},
    "structure": {"(": 0, ")": 1, ".": 2},
    "predicted_loop_type": {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6},
}


def get_couples(structure):
    """
    Parses a dot-bracket structure string to identify paired bases.
    Returns a numpy array where map[i] is the index of the partner of i,
    or -1 if i is unpaired.
    """
    partners = np.full(len(structure), -1, dtype=np.int32)
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


def one_hot_encode(seq, token_map):
    """
    One-hot encodes a sequence string based on the provided token map.
    Returns a numpy array of shape (len(seq), len(token_map)).
    """
    vocab_size = len(token_map)
    encoding = np.zeros((len(seq), vocab_size), dtype=np.float32)
    for i, char in enumerate(seq):
        if char in token_map:
            encoding[i, token_map[char]] = 1.0
    return encoding


class RNADataset(Dataset):
    def __init__(self, inputs, partner_indices, targets=None, ids=None):
        self.inputs = inputs
        self.partner_indices = partner_indices
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # inputs: (SeqLen, Channels)
        # partner_indices: (SeqLen,)
        # targets: (SeqLen, 5)

        item = {
            "inputs": torch.tensor(self.inputs[idx], dtype=torch.float32),
            "partner_indices": torch.tensor(
                self.partner_indices[idx], dtype=torch.long
            ),
        }

        if self.targets is not None:
            item["targets"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        if self.ids is not None:
            item["ids"] = self.ids[idx]

        return item


def preprocess_data(df, mode="train"):
    """
    Generates features and targets from the dataframe.
    """
    # Initialize lists
    all_inputs = []
    all_partner_indices = []
    all_targets = []
    all_ids = df["id"].values

    # Pre-compute token maps size
    seq_vocab_size = len(TOKEN_DICT["sequence"])

    # Iterate over samples
    for idx, row in df.iterrows():
        sequence = row["sequence"]
        structure = row["structure"]
        loop_type = row["predicted_loop_type"]

        seq_len = len(sequence)

        # 1. Base Features
        enc_seq = one_hot_encode(sequence, TOKEN_DICT["sequence"])
        enc_struct = one_hot_encode(structure, TOKEN_DICT["structure"])
        enc_loop = one_hot_encode(loop_type, TOKEN_DICT["predicted_loop_type"])

        # 2. Partner Indices
        partners = get_couples(structure)

        # 3. Partner Identity Feature
        # If i is paired with j, feature is one-hot of sequence[j].
        # If unpaired, feature is all zeros.
        enc_partner = np.zeros((seq_len, seq_vocab_size), dtype=np.float32)
        for i, p_idx in enumerate(partners):
            if p_idx != -1:
                # Get the base at the partner index
                partner_base = sequence[p_idx]
                if partner_base in TOKEN_DICT["sequence"]:
                    enc_partner[i, TOKEN_DICT["sequence"][partner_base]] = 1.0

        # Concatenate all features
        # Shape: (SeqLen, 4 + 3 + 7 + 4) = (SeqLen, 18)
        sample_input = np.concatenate(
            [enc_seq, enc_struct, enc_loop, enc_partner], axis=1
        )

        all_inputs.append(sample_input)
        all_partner_indices.append(partners)

        # 4. Targets (only for train/val)
        if mode in ["train", "val"]:
            sample_targets = []
            for col in Config.TARGET_COLS:
                # Parse stringified list
                try:
                    val_list = ast.literal_eval(row[col])
                    # Pad if necessary (though usually length is seq_scored)
                    # The competition data usually provides 68 values.
                    # We pad to 107 with zeros (masked out by loss anyway).
                    arr = np.array(val_list, dtype=np.float32)
                    pad_len = seq_len - len(arr)
                    if pad_len > 0:
                        arr = np.pad(arr, (0, pad_len), constant_values=0)
                    sample_targets.append(arr)
                except (ValueError, SyntaxError):
                    # Fallback for errors
                    sample_targets.append(np.zeros(seq_len, dtype=np.float32))

            # Stack targets: (SeqLen, 5)
            all_targets.append(np.stack(sample_targets, axis=1))

    # Convert to numpy arrays
    inputs_np = np.array(all_inputs, dtype=np.float32)
    partner_indices_np = np.array(all_partner_indices, dtype=np.int32)

    if mode in ["train", "val"]:
        targets_np = np.array(all_targets, dtype=np.float32)
        return inputs_np, partner_indices_np, targets_np, all_ids
    else:
        return inputs_np, partner_indices_np, None, all_ids


def get_data(mode, load_cached_data=True):
    """
    Retrieves data for the specified mode (train, val, test).
    Uses caching to speed up loading.
    """
    # Determine cache key and file path
    if mode == "train":
        cache_key = Config.CACHE_TRAIN_KEY
        csv_path = Config.TRAIN_METADATA
    elif mode == "val":
        cache_key = Config.CACHE_VAL_KEY
        csv_path = Config.VAL_METADATA
    elif mode == "test":
        cache_key = Config.CACHE_TEST_KEY
        csv_path = Config.TEST_METADATA
    else:
        raise ValueError(f"Invalid mode: {mode}")

    cache_path = Config.get_cache_path(cache_key)

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {mode} data from cache: {cache_path}")
        try:
            data = np.load(cache_path, allow_pickle=True)
            inputs = data["inputs"]
            partner_indices = data["partner_indices"]
            ids = data["ids"]
            targets = data["targets"] if "targets" in data else None

            # If targets was saved as None (test set), np.load might return object array or None
            if str(targets) == "None":
                targets = None

            return inputs, partner_indices, targets, ids
        except Exception as e:
            print(f"Failed to load cache: {e}. Re-processing...")

    # Process from scratch
    print(f"Processing {mode} data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Debug mode: subset data
    if Config.DEBUG:
        df = df.head(Config.MAX_DEBUG_SAMPLES)
        print(f"DEBUG MODE: Reduced {mode} data to {len(df)} samples.")

    inputs, partner_indices, targets, ids = preprocess_data(df, mode=mode)

    # Save to cache
    print(f"Saving {mode} data to cache: {cache_path}")
    save_dict = {"inputs": inputs, "partner_indices": partner_indices, "ids": ids}
    if targets is not None:
        save_dict["targets"] = targets
    else:
        save_dict["targets"] = np.array(None)  # Placeholder

    np.savez_compressed(cache_path, **save_dict)

    return inputs, partner_indices, targets, ids


def get_loader(
    mode, batch_size=None, shuffle=None, num_workers=None, load_cached_data=True
):
    """
    Creates a DataLoader for the specified mode.
    """
    inputs, partner_indices, targets, ids = get_data(mode, load_cached_data)

    dataset = RNADataset(inputs, partner_indices, targets, ids)

    # Defaults
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    if num_workers is None:
        num_workers = Config.NUM_WORKERS
    if shuffle is None:
        shuffle = mode == "train"

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    return loader
