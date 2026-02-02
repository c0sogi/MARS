import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything

# Feature Mapping Constants
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


class RNADataset(Dataset):
    def __init__(self, inputs, targets, partner_indices, ids):
        """
        PyTorch Dataset for RNA Degradation Prediction.

        Args:
            inputs (np.ndarray): Input features of shape (N, 107, 18).
            targets (np.ndarray): Target values of shape (N, 107, 5).
            partner_indices (np.ndarray): Indices of paired bases of shape (N, 107).
                                          Unpaired bases are denoted by -1.
            ids (np.ndarray): Array of sample IDs.
        """
        self.inputs = inputs
        self.targets = targets
        self.partner_indices = partner_indices
        self.ids = ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Convert numpy arrays to PyTorch tensors
        # Input shape: (Seq_Len, Channels) -> (107, 18)
        inp = torch.tensor(self.inputs[idx], dtype=torch.float32)

        # Target shape: (Seq_Len, 5) -> (107, 5)
        # Includes padded zeros for boundary anchoring in the tail (68-107)
        tgt = torch.tensor(self.targets[idx], dtype=torch.float32)

        # Partner Indices: (Seq_Len,) -> (107,)
        # Contains -1 for unpaired bases. Model must handle masking/indexing.
        p_idx = torch.tensor(self.partner_indices[idx], dtype=torch.long)

        return inp, tgt, p_idx


def get_structure_pairs(structure):
    """
    Parses a dot-bracket structure string to identify base pairs.

    Args:
        structure (str): Dot-bracket notation string (e.g., "((..))").

    Returns:
        np.ndarray: Array of length L where arr[i] is the index of the base paired with i.
                    If i is unpaired, arr[i] is -1.
    """
    pairs = np.full(len(structure), -1, dtype=np.int32)
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
    Generates a One-Hot encoding for a sequence based on a mapping dictionary.

    Args:
        seq (str): Input sequence.
        map_dict (dict): Dictionary mapping characters to indices.
        num_classes (int): Total number of classes (channels).

    Returns:
        np.ndarray: One-hot encoded array of shape (Len, num_classes).
    """
    arr = np.zeros((len(seq), num_classes), dtype=np.float32)
    for i, char in enumerate(seq):
        if char in map_dict:
            arr[i, map_dict[char]] = 1.0
    return arr


def preprocess_data(df, mode="train"):
    """
    Processes the dataframe to generate input features and targets.

    Args:
        df (pd.DataFrame): Input dataframe containing sequences and metadata.
        mode (str): 'train', 'val', or 'test'.

    Returns:
        tuple: (inputs, targets, partner_indices, ids)
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LEN

    # Initialize arrays
    # Input Channels: 4 (Seq) + 3 (Struct) + 7 (Loop) + 4 (Partner Identity) = 18
    input_array = np.zeros((num_samples, seq_len, 18), dtype=np.float32)
    target_array = np.zeros((num_samples, seq_len, 5), dtype=np.float32)
    partner_indices_array = np.zeros((num_samples, seq_len), dtype=np.int32)
    ids = np.array(df["id"].tolist())

    # Target columns in the specific order required for the metric/loss
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for idx, row in df.iterrows():
        # --- Feature Engineering ---
        seq = row["sequence"]
        struct = row["structure"]
        loop = row["predicted_loop_type"]

        # 1. Base One-Hot Encodings
        oh_seq = one_hot(seq, SEQ_MAP, 4)
        oh_struct = one_hot(struct, STRUCT_MAP, 3)
        oh_loop = one_hot(loop, LOOP_MAP, 7)

        # 2. Partner Information
        pairs = get_structure_pairs(struct)
        partner_indices_array[idx] = pairs

        # 3. Partner Identity Encoding
        # If base i is paired with j, the partner feature at i is the sequence identity of j.
        # If unpaired, it is a zero vector.
        oh_partner = np.zeros((seq_len, 4), dtype=np.float32)
        for i, j in enumerate(pairs):
            if j != -1:
                oh_partner[i] = oh_seq[j]

        # Concatenate all features
        input_array[idx] = np.concatenate(
            [oh_seq, oh_struct, oh_loop, oh_partner], axis=1
        )

        # --- Target Processing ---
        if mode in ["train", "val"]:
            for t_i, col in enumerate(target_cols):
                val_str = row[col]
                try:
                    # Parse string representation of list
                    vals = ast.literal_eval(val_str)
                except (ValueError, SyntaxError):
                    vals = []

                n_vals = len(vals)
                if n_vals > 0:
                    # Fill the scored positions
                    target_array[idx, :n_vals, t_i] = vals

                # Boundary Anchoring:
                # The remaining positions (n_vals to 107) are left as 0.0.
                # This neutral baseline stabilizes the bidirectional RNN.
        else:
            # Test mode: Targets remain 0.0
            pass

    return input_array, target_array, partner_indices_array, ids


def get_data(csv_path, cache_path, mode="train", load_cached_data=True):
    """
    Loads data from cache if available, otherwise processes from CSV and caches it.

    Args:
        csv_path (str): Path to the source CSV file.
        cache_path (str): Path to the .npz cache file.
        mode (str): Dataset mode ('train', 'val', 'test').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (inputs, targets, partner_indices, ids)
    """
    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {mode} data from {cache_path}...")
        try:
            data = np.load(cache_path, allow_pickle=True)
            return data["inputs"], data["targets"], data["partner_indices"], data["ids"]
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from Scratch
    print(f"Processing {mode} data from {csv_path}...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Source file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    inputs, targets, partner_indices, ids = preprocess_data(df, mode)

    # 3. Save to Cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    print(f"Saving {mode} data to {cache_path}...")
    np.savez_compressed(
        cache_path,
        inputs=inputs,
        targets=targets,
        partner_indices=partner_indices,
        ids=ids,
    )

    return inputs, targets, partner_indices, ids


def get_loaders(load_cached_data=True, batch_size=None):
    """
    Generates DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached .npz files.
        batch_size (int, optional): Batch size override. Defaults to Config.BATCH_SIZE.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    seed_everything(Config.SEED)

    if batch_size is None:
        batch_size = Config.BATCH_SIZE

    # --- Load Data ---
    train_inputs, train_targets, train_p_idx, train_ids = get_data(
        Config.TRAIN_CSV, Config.TRAIN_CACHE, "train", load_cached_data
    )

    val_inputs, val_targets, val_p_idx, val_ids = get_data(
        Config.VAL_CSV, Config.VAL_CACHE, "val", load_cached_data
    )

    test_inputs, test_targets, test_p_idx, test_ids = get_data(
        Config.TEST_CSV, Config.TEST_CACHE, "test", load_cached_data
    )

    # --- Create Datasets ---
    train_dataset = RNADataset(train_inputs, train_targets, train_p_idx, train_ids)
    val_dataset = RNADataset(val_inputs, val_targets, val_p_idx, val_ids)
    test_dataset = RNADataset(test_inputs, test_targets, test_p_idx, test_ids)

    # --- Create Loaders ---
    # Train loader: Shuffle=True, Drop_Last=True for batch stat stability
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # Val/Test loaders: Shuffle=False
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
