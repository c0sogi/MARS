import os
import numpy as np
import pandas as pd
import torch
from library.config import Config

# Define mappings for one-hot encoding
SEQ_MAP = {c: i for i, c in enumerate("AGCU")}
STRUCT_MAP = {c: i for i, c in enumerate("().")}
LOOP_MAP = {c: i for i, c in enumerate("SMIBHEX")}


def get_pair_indices(structure):
    """
    Parses a dot-bracket structure string to find base pair indices.

    Args:
        structure (str): Dot-bracket string (e.g., "((..))").

    Returns:
        np.ndarray: Array of shape (len(structure),) where arr[i] is the index
                    of the base paired with i, or -1 if unpaired.
    """
    n = len(structure)
    indices = np.full(n, -1, dtype=np.int32)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                indices[i] = j
                indices[j] = i

    return indices


def one_hot_encode(sequence, structure, loop_type):
    """
    One-hot encodes sequence, structure, and predicted loop type.

    Args:
        sequence (str): RNA sequence (A, G, C, U).
        structure (str): Dot-bracket structure.
        loop_type (str): Predicted loop type string.

    Returns:
        np.ndarray: Float32 array of shape (SeqLen, 14).
                    Channels 0-3: Sequence
                    Channels 4-6: Structure
                    Channels 7-13: Loop Type
    """
    length = len(sequence)
    # Total channels = 4 (Seq) + 3 (Struct) + 7 (Loop) = 14
    encoding = np.zeros((length, 14), dtype=np.float32)

    for i in range(length):
        # Sequence
        s_char = sequence[i]
        if s_char in SEQ_MAP:
            encoding[i, SEQ_MAP[s_char]] = 1.0

        # Structure
        st_char = structure[i]
        if st_char in STRUCT_MAP:
            encoding[i, 4 + STRUCT_MAP[st_char]] = 1.0

        # Loop Type
        l_char = loop_type[i]
        if l_char in LOOP_MAP:
            encoding[i, 7 + LOOP_MAP[l_char]] = 1.0

    return encoding


class RNAPreprocessor:
    """
    Handles data loading, feature extraction, and caching for the RNA degradation task.
    """

    def __init__(self):
        self.working_dir = Config.WORKING_DIR
        os.makedirs(self.working_dir, exist_ok=True)

    def process_data(self, split="train", load_cached_data=True):
        """
        Loads and processes data for a specific split.

        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): If True, attempts to load from .npz cache.

        Returns:
            dict: Dictionary containing 'ids', 'inputs', 'pair_indices', and optionally 'targets'.
        """
        # Determine file paths based on split
        if split == "train":
            input_path = Config.TRAIN_PATH
            cache_path = os.path.join(self.working_dir, "train_data.npz")
        elif split == "val":
            input_path = Config.VAL_PATH
            cache_path = os.path.join(self.working_dir, "val_data.npz")
        elif split == "test":
            input_path = Config.TEST_PATH
            cache_path = os.path.join(self.working_dir, "test_data.npz")
        else:
            raise ValueError(f"Unknown split: {split}")

        # Attempt to load from cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached data for {split} from {cache_path}")
            try:
                # Allow pickle is needed if object arrays (like strings) are stored,
                # though we aim for numeric arrays.
                with np.load(cache_path, allow_pickle=True) as data:
                    return {key: data[key] for key in data.files}
            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing from scratch.")

        # Process from scratch
        print(f"Processing data for {split} from {input_path}")
        df = pd.read_parquet(input_path)

        # Handle Debug Mode
        if Config.DEBUG:
            print(
                f"DEBUG mode: reducing {split} dataset size to {Config.DEBUG_SUBSET_SIZE}"
            )
            df = df.head(Config.DEBUG_SUBSET_SIZE)

        # Extract raw columns
        ids = df["id"].values
        sequences = df["sequence"].values
        structures = df["structure"].values
        loop_types = df["predicted_loop_type"].values

        n_samples = len(df)
        seq_len = Config.SEQ_LEN

        # Initialize arrays
        # Inputs: (N, SeqLen, Channels)
        inputs = np.zeros((n_samples, seq_len, Config.INPUT_CHANNELS), dtype=np.float32)
        # Pair Indices: (N, SeqLen)
        pair_indices = np.zeros((n_samples, seq_len), dtype=np.int32)

        # Feature Extraction Loop
        for i in range(n_samples):
            inputs[i] = one_hot_encode(sequences[i], structures[i], loop_types[i])
            pair_indices[i] = get_pair_indices(structures[i])

        result = {"ids": ids, "inputs": inputs, "pair_indices": pair_indices}

        # Target Extraction (Train/Val only)
        if split in ["train", "val"]:
            target_cols = Config.TARGET_COLS
            pred_len = Config.PRED_LEN

            # Targets: (N, PredLen, 5)
            # Note: Targets in dataframe are lists of length `pred_len` (68)
            targets = np.zeros(
                (n_samples, pred_len, len(target_cols)), dtype=np.float32
            )

            for idx, col in enumerate(target_cols):
                # Convert column of lists to numpy array
                # We use tolist() then np.array for efficient conversion of object column
                col_data = np.array(df[col].tolist(), dtype=np.float32)
                targets[:, :, idx] = col_data

            result["targets"] = targets

        # Save to cache
        print(f"Saving processed data to {cache_path}")
        np.savez_compressed(cache_path, **result)

        return result
