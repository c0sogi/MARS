import re
import os
import numpy as np
import pandas as pd
import nltk

# Define the list of atoms to track for the auxiliary formula prediction task
# These are the most common elements found in organic chemistry datasets like BMS.
ATOM_LIST = ["C", "H", "N", "O", "S", "F", "Cl", "Br", "I", "B", "P", "Si"]
ATOM_TO_IDX = {atom: i for i, atom in enumerate(ATOM_LIST)}


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking loss and metrics during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_levenshtein(predicted_text, target_text):
    """
    Calculates the Levenshtein distance between two strings.

    Args:
        predicted_text (str): The predicted InChI string.
        target_text (str): The ground truth InChI string.

    Returns:
        int: The edit distance.
    """
    return nltk.edit_distance(predicted_text, target_text)


def parse_molecular_formula(inchi_string):
    """
    Parses an InChI string to extract the counts of specific atoms.

    The function extracts the formula layer (typically index 1 after splitting by '/')
    and uses regex to count atoms defined in ATOM_LIST.

    Args:
        inchi_string (str): The InChI string (e.g., 'InChI=1S/C13H10Cl4O4/c1-20...').

    Returns:
        np.array: A numpy array of shape (len(ATOM_LIST),) containing atom counts.
    """
    counts = np.zeros(len(ATOM_LIST), dtype=np.float32)

    try:
        # The formula is usually the second part of the InChI string
        # e.g., InChI=1S/C13H10Cl4O4/... -> parts[1] is C13H10Cl4O4
        parts = inchi_string.split("/")
        if len(parts) < 2:
            return counts

        formula = parts[1]

        # Regex to find Element followed by optional number
        # Matches: 'C13', 'H10', 'Cl4', 'O', 'N'
        matches = re.findall(r"([A-Z][a-z]?)(\d*)", formula)

        for element, count_str in matches:
            if element in ATOM_TO_IDX:
                # If no number follows the element, the count is 1
                count = int(count_str) if count_str else 1
                counts[ATOM_TO_IDX[element]] += count

    except Exception:
        # In case of parsing error, return zeros (or handle as appropriate)
        pass

    return counts


def get_atom_counts(df, load_cached_data=True, cache_dir="./working/idea_6/"):
    """
    Generates or loads the atom count matrix for a dataframe of InChI strings.
    Implements caching to avoid re-parsing the large dataset.

    Args:
        df (pd.DataFrame): Dataframe containing an 'InChI' column.
        load_cached_data (bool): If True, attempts to load from cache.
        cache_dir (str): Directory to store the cached .npy file.

    Returns:
        np.array: Matrix of shape (len(df), len(ATOM_LIST)) with atom counts.
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Create a unique hash or filename based on the dataframe length to avoid collisions
    # For simplicity in this context, we use a fixed name but ensure directory safety.
    # In a production system, one might hash the dataframe content.
    cache_path = os.path.join(cache_dir, f"atom_counts_{len(df)}.npy")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached atom counts from {cache_path}...")
        try:
            atom_counts = np.load(cache_path)
            # Verify shape matches
            if atom_counts.shape[0] == len(df):
                return atom_counts
            else:
                print("Cached file dimension mismatch. Recomputing...")
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing...")

    print("Parsing molecular formulas from InChI strings (this may take a while)...")

    # Vectorized application is hard with complex regex logic, so we use list comprehension
    # or apply. List comprehension is generally faster for string ops in pandas.
    inchi_series = df["InChI"].astype(str).tolist()

    # Process
    atom_counts_list = [parse_molecular_formula(s) for s in inchi_series]
    atom_counts = np.vstack(atom_counts_list).astype(np.float32)

    # Save to cache
    print(f"Saving atom counts to {cache_path}...")
    np.save(cache_path, atom_counts)

    return atom_counts
