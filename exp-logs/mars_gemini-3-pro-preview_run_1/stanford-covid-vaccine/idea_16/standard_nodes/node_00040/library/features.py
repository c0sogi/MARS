import os
import numpy as np
import pandas as pd
from library.config import Config

# =============================================================================
# Vocabularies
# =============================================================================
# Maps atomic characters to integer indices
SEQ_VOCAB = {"A": 0, "G": 1, "C": 2, "U": 3}
LOOP_VOCAB = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


# =============================================================================
# Helper Functions
# =============================================================================
def get_structure_info(structure, seq_len):
    """
    Parses the dot-bracket structure string to extract geometric and shortcut features.

    Args:
        structure (str): Dot-bracket string (e.g., "..((..))..").
        seq_len (int): Length of the sequence.

    Returns:
        pair_index (np.ndarray): Index of the partner base. If unpaired, points to self.
        pair_dist (np.ndarray): Signed distance (partner - current). 0 if unpaired.
        pair_mask (np.ndarray): 1.0 if paired, 0.0 if unpaired.
    """
    # Initialize defaults
    # If unpaired, pair_index points to self (i), dist is 0, mask is 0.
    pair_index = np.arange(seq_len, dtype=np.int64)
    pair_dist = np.zeros(seq_len, dtype=np.float32)
    pair_mask = np.zeros(seq_len, dtype=np.float32)

    stack = []
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                # Pair formed between j (open) and i (close)

                # For position j (opening): Partner is i (downstream)
                pair_index[j] = i
                pair_dist[j] = float(i - j)  # Positive distance
                pair_mask[j] = 1.0

                # For position i (closing): Partner is j (upstream)
                pair_index[i] = j
                pair_dist[i] = float(j - i)  # Negative distance
                pair_mask[i] = 1.0

    return pair_index, pair_dist, pair_mask


def process_dataframe(df, is_test=False):
    """
    Transforms the DataFrame into a dictionary of numpy arrays suitable for the model.

    Args:
        df (pd.DataFrame): Input dataframe containing sequences and structures.
        is_test (bool): Whether processing test data (skips target extraction).

    Returns:
        dict: Dictionary of numpy arrays (inputs and optional targets).
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LEN

    # Pre-allocate arrays for efficiency
    ids = df["id"].values.astype(str)
    sequence_arr = np.zeros((num_samples, seq_len), dtype=np.int32)
    loop_arr = np.zeros((num_samples, seq_len), dtype=np.int32)
    pair_index_arr = np.zeros((num_samples, seq_len), dtype=np.int64)
    pair_dist_arr = np.zeros((num_samples, seq_len), dtype=np.float32)
    pair_mask_arr = np.zeros((num_samples, seq_len), dtype=np.float32)

    # Prepare targets if training
    targets_arr = None
    if not is_test:
        # Extract the specific scored columns: reactivity, deg_Mg_pH10, deg_Mg_50C
        # Data is stored as lists in the DataFrame (from Parquet)
        try:
            # Stack lists vertically to create (N, 68) arrays
            t_react = np.vstack(df["reactivity"].values)
            t_mg_ph10 = np.vstack(df["deg_Mg_pH10"].values)
            t_mg_50c = np.vstack(df["deg_Mg_50C"].values)

            # Stack along the last dimension to create (N, 68, 3)
            targets_arr = np.stack([t_react, t_mg_ph10, t_mg_50c], axis=-1).astype(
                np.float32
            )
        except Exception as e:
            print(f"Warning: Could not extract targets. {e}")
            targets_arr = None

    # Extract raw values for iteration
    raw_sequences = df["sequence"].values
    raw_structures = df["structure"].values
    raw_loops = df["predicted_loop_type"].values

    for idx in range(num_samples):
        # 1. Encode Sequence
        seq_str = raw_sequences[idx]
        sequence_arr[idx] = [SEQ_VOCAB.get(c, 0) for c in seq_str]

        # 2. Encode Loop Type
        loop_str = raw_loops[idx]
        loop_arr[idx] = [LOOP_VOCAB.get(c, 0) for c in loop_str]

        # 3. Encode Structure (Geometric & Shortcut info)
        struct_str = raw_structures[idx]
        p_idx, p_dist, p_mask = get_structure_info(struct_str, seq_len)
        pair_index_arr[idx] = p_idx
        pair_dist_arr[idx] = p_dist
        pair_mask_arr[idx] = p_mask

    result = {
        "ids": ids,
        "sequence": sequence_arr,
        "loop_type": loop_arr,
        "pair_index": pair_index_arr,
        "pair_dist": pair_dist_arr,
        "pair_mask": pair_mask_arr,
    }

    if targets_arr is not None:
        result["targets"] = targets_arr

    return result


# =============================================================================
# Main Data Loading Function
# =============================================================================
def get_data(split="train", load_cached_data=True):
    """
    Loads data for a specific split, handling caching and processing.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from .npz cache.

    Returns:
        dict: Dictionary containing numpy arrays for inputs and targets.
    """
    # 1. Determine paths based on split
    if split == "train":
        parquet_path = Config.TRAIN_PATH
        cache_path = os.path.join(Config.CACHE_DIR, "train_data.npz")
        is_test = False
    elif split == "val":
        parquet_path = Config.VAL_PATH
        cache_path = os.path.join(Config.CACHE_DIR, "val_data.npz")
        is_test = False
    elif split == "test":
        parquet_path = Config.TEST_PATH
        cache_path = os.path.join(Config.CACHE_DIR, "test_data.npz")
        is_test = True
    else:
        raise ValueError(f"Invalid split: {split}")

    # 2. Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # 3. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {split} data from {cache_path}...")
        try:
            # Load compressed numpy archive
            with np.load(cache_path, allow_pickle=True) as data:
                result = {key: data[key] for key in data.files}

            # Re-insert None for targets if missing (e.g., test set or failed save)
            if "targets" not in result:
                result["targets"] = None

            return result
        except Exception as e:
            print(f"Cache load failed ({e}). Recomputing...")

    # 4. Compute from Scratch
    print(f"Processing {split} data from {parquet_path}...")

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    # Debugging: Subset data if configured
    if Config.DEBUG:
        print(f"DEBUG MODE: Reducing {split} data to {Config.SUBSET_SIZE} samples.")
        df = df.iloc[: Config.SUBSET_SIZE].copy()

    processed_data = process_dataframe(df, is_test=is_test)

    # 5. Save to Cache
    print(f"Saving {split} data to {cache_path}...")
    # Filter out None values before saving to npz to avoid object array issues
    save_dict = {k: v for k, v in processed_data.items() if v is not None}
    np.savez_compressed(cache_path, **save_dict)

    return processed_data
