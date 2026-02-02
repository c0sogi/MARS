import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA Degradation Prediction.
    Implements dynamic input masking for the auxiliary reconstruction task.
    """

    def __init__(self, inputs, targets=None, mode="train", mask_prob=0.15):
        """
        Args:
            inputs (np.ndarray): Input features of shape (N, 107, 14).
            targets (np.ndarray, optional): Target values of shape (N, 107, 5).
            mode (str): 'train', 'val', or 'test'.
            mask_prob (float): Probability of masking a position during training.
        """
        self.inputs = inputs
        self.targets = targets
        self.mode = mode
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Convert input to tensor: Shape (107, 14)
        input_tensor = torch.tensor(self.inputs[idx], dtype=torch.float32)

        # Handle Targets
        if self.targets is not None:
            # Shape (107, 5)
            target_tensor = torch.tensor(self.targets[idx], dtype=torch.float32)
        else:
            # Dummy targets for test set
            target_tensor = torch.zeros(
                (Config.SEQ_LENGTH, Config.OUTPUT_DIM), dtype=torch.float32
            )

        # Dynamic Masking Logic
        if self.mode == "train" and self.mask_prob > 0:
            # Generate a boolean mask: True where we want to mask the input
            # Shape (107,)
            mask_bool = torch.rand(Config.SEQ_LENGTH) < self.mask_prob

            # Create masked input (clone to avoid modifying original)
            masked_input = input_tensor.clone()

            # Apply mask: Set feature vectors at masked positions to zero
            masked_input[mask_bool] = 0.0

            # Convert mask to float for loss calculation (1.0 = masked, 0.0 = unmasked)
            mask_tensor = mask_bool.float()

            # Return: Masked Input (for model), Original Input (for recon target), Regression Targets, Mask
            return masked_input, input_tensor, target_tensor, mask_tensor
        else:
            # Validation/Test: No masking
            mask_tensor = torch.zeros(Config.SEQ_LENGTH, dtype=torch.float32)
            return input_tensor, input_tensor, target_tensor, mask_tensor


def _encode_sequence(seq_str, map_dict, vocab_size):
    """Helper to one-hot encode a sequence string."""
    indices = [map_dict.get(char, 0) for char in seq_str]
    one_hot = np.zeros((len(seq_str), vocab_size), dtype=np.float32)
    one_hot[np.arange(len(seq_str)), indices] = 1.0
    return one_hot


def process_dataframe(df, has_targets=True):
    """
    Converts DataFrame columns into numpy arrays for inputs and targets.
    """
    # Dictionaries for One-Hot Encoding
    seq_map = {"A": 0, "G": 1, "C": 2, "U": 3}
    struct_map = {".": 0, "(": 1, ")": 2}
    loop_map = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

    num_samples = len(df)
    seq_len = Config.SEQ_LENGTH
    input_dim = Config.INPUT_DIM

    # Pre-allocate input array: (N, 107, 14)
    inputs = np.zeros((num_samples, seq_len, input_dim), dtype=np.float32)

    # Pre-allocate target array: (N, 107, 5) if targets exist
    if has_targets:
        targets = np.zeros((num_samples, seq_len, Config.OUTPUT_DIM), dtype=np.float32)
    else:
        targets = None

    for i, row in df.iterrows():
        # 1. Encode Sequence (4 dims)
        seq_oh = _encode_sequence(row["sequence"], seq_map, Config.VOCAB_SIZE_SEQ)

        # 2. Encode Structure (3 dims)
        struct_oh = _encode_sequence(
            row["structure"], struct_map, Config.VOCAB_SIZE_STRUCT
        )

        # 3. Encode Loop Type (7 dims)
        loop_oh = _encode_sequence(
            row["predicted_loop_type"], loop_map, Config.VOCAB_SIZE_LOOP
        )

        # Concatenate features along the channel dimension
        inputs[i] = np.concatenate([seq_oh, struct_oh, loop_oh], axis=1)

        # 4. Process Targets
        if has_targets:
            # Targets are provided as lists/arrays of length `seq_scored` (68)
            # We pad them to `seq_length` (107) with zeros.
            for t_idx, col in enumerate(Config.TARGET_COLS):
                val_list = row[col]
                if isinstance(val_list, (list, np.ndarray)):
                    length = len(val_list)
                    # Copy available data
                    targets[i, :length, t_idx] = val_list
                    # Remaining positions are already 0.0 from initialization

    return inputs, targets


def load_and_preprocess_data(data_path, split_name, load_cached_data=True):
    """
    Loads data from Parquet, preprocesses it into numpy arrays, and manages caching.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(Config.CACHE_DIR, f"{split_name}_data.npz")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading {split_name} data from cache: {cache_file}")
        try:
            data = np.load(cache_file)
            inputs = data["inputs"]
            if "targets" in data:
                targets = data["targets"]
                # Handle case where targets might be saved as a 0-d array (None)
                if targets.ndim == 0:
                    targets = None
            else:
                targets = None
            return inputs, targets
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Processing {split_name} data from {data_path}...")
    df = pd.read_parquet(data_path)

    # Check if target columns exist (Test set won't have them)
    has_targets = all(col in df.columns for col in Config.TARGET_COLS)

    inputs, targets = process_dataframe(df, has_targets=has_targets)

    # 3. Save to cache
    print(f"Saving {split_name} data to cache: {cache_file}")
    if targets is not None:
        np.savez(cache_file, inputs=inputs, targets=targets)
    else:
        np.savez(cache_file, inputs=inputs)

    return inputs, targets


def get_dataloaders(
    train_batch_size=Config.BATCH_SIZE,
    val_batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
    debug=False,
):
    """
    Main entry point to get DataLoaders for Train, Val, and Test.
    """
    # Load and process data
    train_inputs, train_targets = load_and_preprocess_data(
        Config.TRAIN_DATA_PATH, "train", load_cached_data
    )
    val_inputs, val_targets = load_and_preprocess_data(
        Config.VAL_DATA_PATH, "val", load_cached_data
    )
    test_inputs, test_targets = load_and_preprocess_data(
        Config.TEST_DATA_PATH, "test", load_cached_data
    )

    # Debugging: Use a small subset
    if debug:
        print("DEBUG MODE: Using subset of data.")
        subset = 100
        train_inputs = train_inputs[:subset]
        train_targets = train_targets[:subset]
        val_inputs = val_inputs[:subset]
        val_targets = val_targets[:subset]
        # Test set is small enough, but can subset if needed
        test_inputs = test_inputs[:subset]
        if test_targets is not None:
            test_targets = test_targets[:subset]

    # Initialize Datasets
    # Train: Enable masking
    train_dataset = RNADataset(
        train_inputs, train_targets, mode="train", mask_prob=Config.MASK_PROB
    )
    # Val/Test: Disable masking
    val_dataset = RNADataset(val_inputs, val_targets, mode="val", mask_prob=0.0)
    test_dataset = RNADataset(test_inputs, test_targets, mode="test", mask_prob=0.0)

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(Config.DEVICE == "cuda"),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    return train_loader, val_loader, test_loader
