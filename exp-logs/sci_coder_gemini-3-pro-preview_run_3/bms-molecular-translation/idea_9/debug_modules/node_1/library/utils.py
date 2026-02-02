import os
import random
import numpy as np
import torch
import re
import pandas as pd
import nltk


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across various libraries.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LevenshteinMetric:
    """
    Computes the mean Levenshtein distance between predictions and targets.
    """

    def __init__(self):
        self.total_distance = 0
        self.count = 0

    def update(self, preds, targets):
        """
        Update the metric with a batch of predictions and targets.

        Args:
            preds (list of str): List of predicted InChI strings.
            targets (list of str): List of ground truth InChI strings.
        """
        for p, t in zip(preds, targets):
            # nltk.edit_distance calculates Levenshtein distance
            dist = nltk.edit_distance(p, t)
            self.total_distance += dist
            self.count += 1

    def compute(self):
        """
        Compute the average Levenshtein distance.
        """
        if self.count == 0:
            return 0.0
        return self.total_distance / self.count

    def reset(self):
        """
        Reset the internal state of the metric.
        """
        self.total_distance = 0
        self.count = 0


# Standard list of atoms expected in the organic molecules for this task
ATOM_VOCAB = ["B", "Br", "C", "Cl", "F", "H", "I", "N", "O", "P", "S", "Si"]
ATOM_TO_IDX = {atom: i for i, atom in enumerate(ATOM_VOCAB)}


def get_atom_counts(inchi_str):
    """
    Parses a single InChI string and extracts atom counts based on the formula layer.

    Args:
        inchi_str (str): The InChI string (e.g., 'InChI=1S/C6H12O6/...')

    Returns:
        np.ndarray: A float32 array of shape (len(ATOM_VOCAB),) containing counts.
    """
    counts = np.zeros(len(ATOM_VOCAB), dtype=np.float32)

    # InChI strings start with "InChI=1S/" followed by the formula layer
    # Example: InChI=1S/C13H10Cl4O4/c1-20...
    try:
        parts = inchi_str.split("/")
        if len(parts) >= 2:
            formula = parts[1]

            # Regex to find Element symbol (e.g., C, Cl) and optional count (e.g., 13, '')
            # [A-Z][a-z]? matches one uppercase followed optionally by one lowercase
            # \d* matches zero or more digits
            matches = re.findall(r"([A-Z][a-z]?)(\d*)", formula)

            for element, count_str in matches:
                if element in ATOM_TO_IDX:
                    # If count number is missing, it implies 1 (e.g., "CH4" -> C1, H4)
                    count = int(count_str) if count_str else 1
                    counts[ATOM_TO_IDX[element]] += count
    except Exception:
        # In case of malformed string, return zeros (or handle as needed)
        pass

    return counts


def compute_and_cache_atom_counts(metadata_path, load_cached_data=True, max_rows=None):
    """
    Computes atom counts for all samples in the metadata file and caches the result.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        load_cached_data (bool): If True, attempts to load from cache first.
        max_rows (int, optional): Limit number of rows to process (for debugging).

    Returns:
        np.ndarray: Array of atom counts for the dataset.
    """
    # Ensure working directory exists
    cache_dir = "./working/idea_9/"
    os.makedirs(cache_dir, exist_ok=True)

    # Create a unique cache filename based on metadata name and max_rows
    base_name = os.path.basename(metadata_path).replace(".csv", "")
    suffix = f"_{max_rows}" if max_rows is not None else ""
    cache_path = os.path.join(cache_dir, f"{base_name}_atom_counts{suffix}.npy")

    # 1. Try to load from cache
    if load_cached_data:
        if os.path.exists(cache_path):
            print(f"Loading cached atom counts from {cache_path}")
            try:
                return np.load(cache_path)
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")
        else:
            print(f"Cache not found at {cache_path}. Computing...")

    # 2. Compute from scratch
    print(f"Computing atom counts for {metadata_path}...")
    df = pd.read_csv(metadata_path)

    if max_rows is not None:
        df = df.iloc[:max_rows]

    if "InChI" not in df.columns:
        print("Warning: 'InChI' column not found in metadata. Returning zero vectors.")
        return np.zeros((len(df), len(ATOM_VOCAB)), dtype=np.float32)

    # Process
    all_counts = []
    # Iterate and parse
    for inchi in df["InChI"]:
        all_counts.append(get_atom_counts(str(inchi)))

    result = np.vstack(all_counts)

    # Save to cache
    print(f"Saving atom counts to {cache_path}")
    np.save(cache_path, result)

    return result
