import os
import random
import numpy as np
import torch
import gc
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def jaccard(str1, str2):
    """
    Computes the Word-level Jaccard score between two strings.
    Based on the implementation provided in the task description.

    Args:
        str1 (str): The ground truth string.
        str2 (str): The predicted string.

    Returns:
        float: The Jaccard similarity score.
    """
    # Handle edge cases where inputs might be non-string (e.g. NaN in pandas)
    if not isinstance(str1, str):
        str1 = ""
    if not isinstance(str2, str):
        str2 = ""

    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)

    denominator = len(a) + len(b) - len(c)
    if denominator == 0:
        return 0.0

    return float(len(c)) / denominator


def compute_average_jaccard(ground_truths, predictions):
    """
    Computes the average Jaccard score for a list of ground truths and predictions.

    Args:
        ground_truths (list): List of ground truth answer strings.
        predictions (list): List of predicted answer strings.

    Returns:
        float: The average Jaccard score.
    """
    if not ground_truths or not predictions:
        return 0.0

    if len(ground_truths) != len(predictions):
        raise ValueError(
            f"Mismatch in number of samples: {len(ground_truths)} vs {len(predictions)}"
        )

    scores = [jaccard(gt, dt) for gt, dt in zip(ground_truths, predictions)]
    return sum(scores) / len(scores)


def cleanup():
    """
    Forces garbage collection and empties the CUDA cache to free up memory.
    Useful to call between folds or after heavy inference steps.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
