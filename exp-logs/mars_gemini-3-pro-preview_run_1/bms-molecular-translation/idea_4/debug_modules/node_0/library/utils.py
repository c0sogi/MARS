import re
import numpy as np
import nltk
from library.config import Config


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking loss and other metrics during training.
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


def compute_levenshtein(prediction, target):
    """
    Computes the Levenshtein distance between a prediction string and a target string.

    Args:
        prediction (str): The predicted InChI string.
        target (str): The ground truth InChI string.

    Returns:
        int: The Levenshtein edit distance.
    """
    return nltk.edit_distance(prediction, target)


def parse_inchi_attributes(inchi_str):
    """
    Parses an InChI string to extract atom counts and total string length.

    This function extracts the chemical formula layer from the InChI string
    (typically the segment after 'InChI=1S/') and counts the occurrences of
    specific atoms defined in Config.ATOM_KEYS. It also calculates the total
    length of the InChI string.

    Args:
        inchi_str (str): The InChI text string (e.g., 'InChI=1S/C13H20OS/...').

    Returns:
        np.ndarray: A 1D numpy array of type float32 and length Config.ATTRIBUTE_DIM.
                    The first N elements correspond to the counts of atoms in
                    Config.ATOM_KEYS. The last element is the length of the inchi_str.
    """
    # Initialize counts for the atoms of interest
    atom_counts = {atom: 0 for atom in Config.ATOM_KEYS}

    # The standard InChI format starts with "InChI=1S/" followed by the formula layer.
    # We split by '/' to isolate the layers.
    parts = inchi_str.split("/")

    # The formula is typically the second part (index 1)
    if len(parts) > 1:
        formula = parts[1]

        # Regex to match ElementSymbol followed by optional Count
        # e.g., "C13" -> ("C", "13"), "H20" -> ("H", "20"), "O" -> ("O", "")
        # [A-Z][a-z]? matches chemical symbols (e.g., C, Cl, N)
        # \d* matches the count (empty if 1)
        matches = re.findall(r"([A-Z][a-z]?)(\d*)", formula)

        for element, count_str in matches:
            # If count is missing, it implies 1
            count = int(count_str) if count_str else 1

            # We only record atoms that are in our predefined list
            if element in atom_counts:
                atom_counts[element] += count

    # Construct the output vector
    # Order must match Config.ATOM_KEYS
    attributes = [atom_counts[key] for key in Config.ATOM_KEYS]

    # Append the total string length as the final attribute
    attributes.append(len(inchi_str))

    return np.array(attributes, dtype=np.float32)
