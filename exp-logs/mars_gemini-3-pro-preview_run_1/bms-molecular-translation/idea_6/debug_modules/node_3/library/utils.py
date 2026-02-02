import os
import re
import numpy as np
import pandas as pd
import nltk
from library.config import Config


def extract_attributes(inchi_str):
    """
    Parses an InChI string to extract atom counts and sequence length.

    Args:
        inchi_str (str): The InChI string (e.g., 'InChI=1S/C6H12O6/...').

    Returns:
        np.ndarray: A 1D array of shape (NUM_ATTRIBUTES,) containing atom counts
                    and the total string length.
    """
    # Initialize counts for keys defined in Config
    atom_counts = {key: 0 for key in Config.ATOM_KEYS}

    # Extract Formula Layer
    # Standard InChI starts with "InChI=1S/" followed by the formula layer at index 1
    try:
        parts = inchi_str.split("/")
        if len(parts) > 1:
            formula = parts[1]
            # Regex to find Element + Count pairs
            # Matches 'C13', 'H20', 'O', 'S' etc.
            # Group 1 is the element symbol, Group 2 is the count (empty string means 1)
            matches = re.findall(r"([A-Z][a-z]?)(\d*)", formula)

            for element, count_str in matches:
                if element in atom_counts:
                    count = int(count_str) if count_str else 1
                    atom_counts[element] += count
    except Exception:
        # In case of malformed strings, we return zeros for atoms but still count length
        pass

    # Create the attribute vector based on the fixed order in Config
    attributes = [atom_counts[k] for k in Config.ATOM_KEYS]

    # Append the sequence length of the entire InChI string
    attributes.append(len(inchi_str))

    return np.array(attributes, dtype=np.float32)


def compute_attribute_stats(df=None, load_cached_data=True):
    """
    Computes or loads the mean and standard deviation of attributes
    from the training set for Z-score normalization.

    Implements deterministic caching using .npy files.

    Args:
        df (pd.DataFrame, optional): DataFrame containing 'InChI' column.
                                     If None, loads from Config.TRAIN_METADATA_PATH.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (mean, std) where each is a np.ndarray of shape (NUM_ATTRIBUTES,).
    """
    stats_path = Config.ATTR_STATS_PATH

    # Ensure working directory exists
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(stats_path):
        print(f"Loading attribute stats from {stats_path}")
        try:
            # We saved as a stacked array [mean, std] to avoid pickle
            stats = np.load(stats_path)
            mean = stats[0]
            std = stats[1]
            return mean, std
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print("Computing attribute statistics from training data...")

    # Load data if not provided
    if df is None:
        print(f"Loading training metadata from {Config.TRAIN_METADATA_PATH}")
        df = pd.read_csv(Config.TRAIN_METADATA_PATH)

    # Extract attributes for all samples
    # We use a list comprehension for efficiency
    inchi_values = df["InChI"].astype(str).values
    all_attrs = [extract_attributes(s) for s in inchi_values]
    all_attrs = np.vstack(all_attrs)

    # Calculate global mean and std
    mean = np.mean(all_attrs, axis=0)
    std = np.std(all_attrs, axis=0)

    # Handle constant attributes (std=0) to avoid division by zero during normalization
    # Setting std to 1.0 preserves the mean offset without scaling
    std[std == 0] = 1.0

    # 3. Save to cache
    # Stack mean and std to save as a single numpy array (2, NUM_ATTRIBUTES)
    # This avoids using pickle which is required for np.save on dicts
    stacked_stats = np.vstack([mean, std])
    np.save(stats_path, stacked_stats)
    print(f"Attribute stats saved to {stats_path}")

    return mean, std


def compute_levenshtein(predictions, ground_truths):
    """
    Calculates the mean Levenshtein distance between predictions and ground truths.

    Args:
        predictions (list of str): List of predicted InChI strings.
        ground_truths (list of str): List of actual InChI strings.

    Returns:
        float: The mean Levenshtein distance.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"Predictions ({len(predictions)}) and ground truths ({len(ground_truths)}) must have the same length."
        )

    distances = []
    for pred, truth in zip(predictions, ground_truths):
        # nltk.edit_distance computes Levenshtein distance
        dist = nltk.edit_distance(pred, truth)
        distances.append(dist)

    return np.mean(distances)


def normalize_attributes(attributes, mean, std):
    """
    Applies Z-score normalization to attributes.

    Args:
        attributes (np.ndarray or torch.Tensor): Raw attributes.
        mean (np.ndarray or torch.Tensor): Mean vector.
        std (np.ndarray or torch.Tensor): Standard deviation vector.

    Returns:
        Normalized attributes.
    """
    return (attributes - mean) / std


def denormalize_attributes(normalized_attributes, mean, std):
    """
    Reverses Z-score normalization.

    Args:
        normalized_attributes (np.ndarray or torch.Tensor): Normalized attributes.
        mean (np.ndarray or torch.Tensor): Mean vector.
        std (np.ndarray or torch.Tensor): Standard deviation vector.

    Returns:
        Raw attributes (approximate).
    """
    return (normalized_attributes * std) + mean
