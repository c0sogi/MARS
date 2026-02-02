import os
import re
import numpy as np
import pandas as pd
from library.config import Config


def parse_inchi_stoichiometry(inchi_string):
    """
    Parses an InChI string to extract the counts of atoms in the molecular formula.

    The function extracts the formula layer from the InChI string (typically the
    second segment delimited by '/') and uses regular expressions to count
    occurrences of each element.

    Args:
        inchi_string (str): The InChI string (e.g., 'InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H').

    Returns:
        dict: A dictionary mapping atom symbols (str) to their integer counts.
              Returns an empty dictionary if parsing fails.
    """
    if not isinstance(inchi_string, str):
        return {}

    # Standard InChI format: InChI=1S/<formula>/...
    # We split by '/' and take the second part (index 1) as the formula.
    try:
        parts = inchi_string.split("/")
        if len(parts) < 2:
            return {}
        formula = parts[1]
    except Exception:
        return {}

    # Regex to identify Element (Capital followed by optional lowercase) and Count (digits)
    # Examples: C6, H12, O, Cl2.
    # Group 1: Symbol, Group 2: Count (empty string implies 1)
    pattern = r"([A-Z][a-z]?)(\d*)"
    matches = re.findall(pattern, formula)

    atom_counts = {}
    for atom, count_str in matches:
        count = int(count_str) if count_str else 1
        atom_counts[atom] = atom_counts.get(atom, 0) + count

    return atom_counts


def get_atom_vector(inchi_string):
    """
    Converts an InChI string into a numerical vector representing atom counts
    for the specific atoms defined in Config.ATOM_LIST.

    Args:
        inchi_string (str): The InChI string.

    Returns:
        np.ndarray: A float32 array of length Config.NUM_ATOMS containing the counts.
    """
    counts = parse_inchi_stoichiometry(inchi_string)
    vector = np.zeros(Config.NUM_ATOMS, dtype=np.float32)

    for i, atom in enumerate(Config.ATOM_LIST):
        vector[i] = counts.get(atom, 0.0)

    return vector


def process_labels(inchi_list, load_cached_data=True):
    """
    Processes a list of InChI strings into a matrix of atom count vectors.
    Implements caching to disk (npy format) to avoid re-computation on subsequent runs.

    Args:
        inchi_list (list or pd.Series): List of InChI strings to process.
        load_cached_data (bool): If True, attempts to load from disk cache first.

    Returns:
        np.ndarray: A matrix of shape (N, NUM_ATOMS) containing atom counts.
    """
    cache_path = Config.TRAIN_LABELS_CACHE_PATH

    # Ensure working directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached label vectors from {cache_path}...")
        try:
            data = np.load(cache_path)
            # Basic validation: check if length matches input
            if len(data) == len(inchi_list):
                return data
            else:
                print(
                    f"Cache size mismatch (Cache: {len(data)}, Input: {len(inchi_list)}). Recomputing..."
                )
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print("Processing InChI labels to generate atom vectors...")

    # Convert to list if it's a pandas Series for efficient iteration
    if isinstance(inchi_list, pd.Series):
        inchi_list = inchi_list.tolist()

    n_samples = len(inchi_list)
    atom_matrix = np.zeros((n_samples, Config.NUM_ATOMS), dtype=np.float32)

    for idx, inchi in enumerate(inchi_list):
        atom_matrix[idx] = get_atom_vector(inchi)

    # 3. Save to cache
    print(f"Saving label vectors to {cache_path}...")
    try:
        np.save(cache_path, atom_matrix)
    except Exception as e:
        print(f"Warning: Failed to save cache to {cache_path}: {e}")

    return atom_matrix
