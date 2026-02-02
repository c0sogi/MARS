import os
import random
import numpy as np
import torch
from nltk.metrics import edit_distance
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_levenshtein(predictions, targets):
    """
    Computes the mean Levenshtein distance between a list of predictions and targets.

    Args:
        predictions (list[str]): A list of predicted InChI strings.
        targets (list[str]): A list of ground truth InChI strings.

    Returns:
        float: The mean Levenshtein distance.

    Raises:
        ValueError: If the lengths of predictions and targets do not match.
    """
    if len(predictions) != len(targets):
        raise ValueError(
            f"Predictions and targets must have the same length. "
            f"Got predictions={len(predictions)}, targets={len(targets)}."
        )

    # Calculate Levenshtein distance for each pair
    distances = [edit_distance(p, t) for p, t in zip(predictions, targets)]

    # Return the mean distance
    return float(np.mean(distances))
