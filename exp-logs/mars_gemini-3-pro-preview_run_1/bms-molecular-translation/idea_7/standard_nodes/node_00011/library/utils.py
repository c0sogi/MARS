import re
import os
import numpy as np
import pandas as pd
from library.config import Config, seed_everything


def parse_inchi_attributes(inchi_str):
    """
    Parses an InChI string to extract atom counts and sequence length.
    Target attributes: C, H, O, N, S, Halogens (F, Cl, Br, I), Length.

    Args:
        inchi_str (str): The InChI string to parse.

    Returns:
        np.ndarray: A float32 array of shape (7,) containing counts and length.
    """
    # Ensure input is a string
    inchi_str = str(inchi_str)

    # Initialize counts map. Order corresponds to Config.ATTR_COLS:
    # ["C", "H", "O", "N", "S", "Halogen", "Length"]
    counts = {"C": 0, "H": 0, "O": 0, "N": 0, "S": 0, "Halogen": 0}

    # Extract formula part. Standard format: InChI=1S/Formula/c...
    # The formula is typically the segment at index 1 after splitting by '/'
    try:
        segments = inchi_str.split("/")
        if len(segments) > 1:
            formula = segments[1]

            # Parse formula using regex
            # Matches Element symbol (e.g., C, Cl) and optional count digits
            matches = re.findall(r"([A-Z][a-z]?)(\d*)", formula)

            for elem, count_str in matches:
                # If count digit is missing, it implies 1
                count = int(count_str) if count_str else 1

                if elem in ["C", "H", "O", "N", "S"]:
                    counts[elem] += count
                elif elem in ["F", "Cl", "Br", "I"]:
                    counts["Halogen"] += count
                # Other elements are ignored for this specific attribute set
    except Exception:
        # In case of malformed string, we return zeros for counts but keep length
        pass

    # Total length of the string
    length = len(inchi_str)

    return np.array(
        [
            counts["C"],
            counts["H"],
            counts["O"],
            counts["N"],
            counts["S"],
            counts["Halogen"],
            length,
        ],
        dtype=np.float32,
    )


def compute_attribute_stats(train_df=None, load_cached_data=True):
    """
    Computes (or loads) the mean and standard deviation of attributes
    from the training set for Z-score normalization.

    Args:
        train_df (pd.DataFrame, optional): Training dataframe containing 'InChI'.
                                           If None, loads from Config.TRAIN_METADATA.
        load_cached_data (bool): Whether to try loading from cache.

    Returns:
        np.ndarray: shape (2, NUM_ATTRIBUTES). Row 0 is mean, Row 1 is std.
    """
    cache_path = Config.ATTR_STATS_PATH
    cache_dir = os.path.dirname(cache_path)

    # Ensure directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading attribute stats from {cache_path}")
            stats = np.load(cache_path)
            return stats
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print("Computing attribute stats from training data...")

    if train_df is None:
        if os.path.exists(Config.TRAIN_METADATA):
            train_df = pd.read_csv(Config.TRAIN_METADATA)
        else:
            raise FileNotFoundError(
                f"Training metadata not found at {Config.TRAIN_METADATA}"
            )

    # Parse attributes for all rows
    # This applies the parsing logic to the entire training set
    attributes = train_df["InChI"].apply(parse_inchi_attributes)

    # Stack into a matrix (N_samples, NUM_ATTRIBUTES)
    attr_matrix = np.stack(attributes.values)

    # Compute Mean and Std along axis 0 (across samples)
    means = np.mean(attr_matrix, axis=0)
    stds = np.std(attr_matrix, axis=0)

    # Handle zero std (constant columns) to avoid division by zero during normalization
    # Replace 0 with 1.0 so division doesn't change the value (which is 0 after mean subtraction)
    stds[stds == 0] = 1.0

    stats = np.vstack([means, stds])

    # 3. Save to cache
    print(f"Saving attribute stats to {cache_path}")
    np.save(cache_path, stats)

    return stats
