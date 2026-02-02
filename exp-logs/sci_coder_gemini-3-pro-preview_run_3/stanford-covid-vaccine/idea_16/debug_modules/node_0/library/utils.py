import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mcrmse_loss(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    Args:
        y_true (torch.Tensor): Ground truth tensor of shape (Batch, Seq_Len, Targets).
        y_pred (torch.Tensor): Predicted tensor of shape (Batch, Seq_Len, Targets).

    Returns:
        torch.Tensor: Scalar loss value.
    """
    # Calculate Mean Squared Error per target column
    # Average over Batch (dim 0) and Sequence (dim 1)
    # y_true and y_pred are expected to be sliced to the scored sequence length before passing here
    mse = torch.mean((y_true - y_pred) ** 2, dim=(0, 1))

    # Calculate Root Mean Squared Error per target column
    rmse = torch.sqrt(mse)

    # Calculate Mean of RMSEs across targets
    loss = torch.mean(rmse)

    return loss


def parse_structure_to_indices(structure):
    """
    Parses a dot-bracket structure string into an array of paired indices.

    Args:
        structure (str): Dot-bracket notation string (e.g., "((..))").

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


def _one_hot_encode_sequence(sequence, token_map, length):
    """
    Helper to one-hot encode a single sequence string.

    Args:
        sequence (str): The sequence to encode.
        token_map (dict): Mapping from character to channel index.
        length (int): Desired length of the output encoding.

    Returns:
        np.ndarray: One-hot encoded array of shape (length, len(token_map)).
    """
    num_tokens = len(token_map)
    encoding = np.zeros((length, num_tokens), dtype=np.float32)

    for i, char in enumerate(sequence):
        if i >= length:
            break
        if char in token_map:
            encoding[i, token_map[char]] = 1.0

    return encoding


def preprocess_data(split_name, load_cached_data=True):
    """
    Loads and preprocesses data for a specific split (train, val, test).
    Implements caching to disk using .npz format to avoid re-processing.

    Args:
        split_name (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        dict: Dictionary containing 'ids', 'features', 'pair_indices', and optionally 'targets'.
    """
    # Validate split_name
    if split_name not in ["train", "val", "test"]:
        raise ValueError(
            f"Invalid split_name: {split_name}. Must be 'train', 'val', or 'test'."
        )

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache path
    cache_path = os.path.join(Config.WORKING_DIR, f"{split_name}_data_cache.npz")

    # Attempt to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            loaded = np.load(cache_path)
            # Reconstruct dictionary from NpzFile
            data_dict = {key: loaded[key] for key in loaded.files}
            return data_dict
        except Exception:
            pass  # Fallback to processing if load fails

    # Load raw data from metadata parquet files
    if split_name == "train":
        file_path = Config.TRAIN_DATA_PATH
    elif split_name == "val":
        file_path = Config.VAL_DATA_PATH
    else:
        file_path = Config.TEST_DATA_PATH

    df = pd.read_parquet(file_path)

    # Extract columns
    ids = df["id"].values
    sequences = df["sequence"].values
    structures = df["structure"].values
    loops = df["predicted_loop_type"].values

    n_samples = len(df)
    seq_len = Config.SEQ_LENGTH

    # Initialize arrays
    # Features: (N, L, C) - Concatenated One-Hot Encodings
    features = np.zeros((n_samples, seq_len, Config.INPUT_CHANNELS), dtype=np.float32)
    # Pair Indices: (N, L) - For structural injection
    pair_indices = np.zeros((n_samples, seq_len), dtype=np.int32)

    # Process each sample
    for i in range(n_samples):
        # 1. One-hot encode sequence (4 channels)
        seq_oh = _one_hot_encode_sequence(sequences[i], Config.TOKEN2INT_SEQ, seq_len)

        # 2. One-hot encode structure (3 channels)
        struct_oh = _one_hot_encode_sequence(
            structures[i], Config.TOKEN2INT_STRUCT, seq_len
        )

        # 3. One-hot encode loop type (7 channels)
        loop_oh = _one_hot_encode_sequence(loops[i], Config.TOKEN2INT_LOOP, seq_len)

        # Concatenate features along channel dimension
        features[i] = np.concatenate([seq_oh, struct_oh, loop_oh], axis=1)

        # 4. Parse structure indices
        pair_indices[i] = parse_structure_to_indices(structures[i])

    data_dict = {"ids": ids, "features": features, "pair_indices": pair_indices}

    # Process targets for train/val splits
    if split_name in ["train", "val"]:
        # Targets shape: (N, Seq_Scored, Num_Targets)
        # Note: Targets are only provided for the first SEQ_SCORED (68) positions
        targets = np.zeros(
            (n_samples, Config.SEQ_SCORED, Config.NUM_TARGETS), dtype=np.float32
        )

        for t_idx, col in enumerate(Config.TARGET_COLS):
            # The parquet file stores these columns as lists/arrays.
            # We convert the column (Series of lists) to a 2D numpy array.
            # We use tolist() to convert the Series to a list of lists, then np.array.
            col_values = np.array(df[col].tolist())
            targets[:, :, t_idx] = col_values

        data_dict["targets"] = targets

    # Save to cache
    np.savez(cache_path, **data_dict)

    return data_dict
