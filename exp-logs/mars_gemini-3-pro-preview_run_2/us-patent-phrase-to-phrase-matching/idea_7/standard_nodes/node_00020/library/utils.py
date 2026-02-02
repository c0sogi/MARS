import os
import random
import numpy as np
import torch
from scipy.stats import pearsonr


def seed_everything(seed=42):
    """
    Seeds all random number generators to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_score(y_true, y_pred):
    """
    Computes the Pearson correlation coefficient between true and predicted scores.

    Args:
        y_true: Array-like of ground truth scores.
        y_pred: Array-like of predicted scores.

    Returns:
        float: The Pearson correlation coefficient.
    """
    # Ensure inputs are flattened numpy arrays to handle various tensor/list shapes
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # pearsonr returns (statistic, p-value); we return the statistic
    score, _ = pearsonr(y_true, y_pred)
    return score


class CPCDict(dict):
    """
    A custom dictionary that maps CPC codes to descriptions.
    Implements a fallback mechanism: if a specific subclass code (e.g., 'A47')
    is not found, it returns the description of the parent Section (e.g., 'A').
    """

    def __missing__(self, key):
        if isinstance(key, str) and len(key) > 0:
            # Fallback to section level (first char)
            section = key[0]
            if section in self:
                return self[section]
        return ""

    def get(self, key, default=None):
        # Check if key exists directly
        if key in self:
            return super().get(key)

        # Fallback logic for get() method
        if isinstance(key, str) and len(key) > 0:
            section = key[0]
            if section in self:
                return self[section]

        return default


def get_cpc_texts():
    """
    Returns a dictionary mapping CPC context codes to their full textual descriptions.
    Populated with the main CPC Section headers to provide domain grounding.

    Returns:
        CPCDict: Dictionary with smart fallback for subclass codes.
    """
    cpc_data = {
        "A": "Human Necessities",
        "B": "Performing Operations; Transporting",
        "C": "Chemistry; Metallurgy",
        "D": "Textiles; Paper",
        "E": "Fixed Constructions",
        "F": "Mechanical Engineering; Lighting; Heating; Weapons; Blasting",
        "G": "Physics",
        "H": "Electricity",
        "Y": "General Tagging of New Technological Developments",
    }
    return CPCDict(cpc_data)
