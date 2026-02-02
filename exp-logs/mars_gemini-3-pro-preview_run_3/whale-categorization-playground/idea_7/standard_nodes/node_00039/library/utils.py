import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_map5(predictions, targets):
    """
    Calculates the Mean Average Precision @ 5 (MAP@5).

    The metric is calculated as the mean of (1 / rank) where rank is the
    position of the correct label in the top 5 predictions (1-indexed).
    If the correct label is not in the top 5, the score for that sample is 0.

    Args:
        predictions (list or np.ndarray): A list (or array) where each element is a list
                                          of up to 5 predicted labels/ids for a sample.
        targets (list or np.ndarray): A list (or array) of the ground truth labels/ids.

    Returns:
        float: The calculated MAP@5 score.
    """
    # Convert numpy arrays to lists if necessary for easier iteration
    if isinstance(predictions, np.ndarray):
        predictions = predictions.tolist()
    if isinstance(targets, np.ndarray):
        targets = targets.tolist()

    n_samples = len(targets)
    if n_samples == 0:
        return 0.0

    total_score = 0.0

    for preds, target in zip(predictions, targets):
        # Consider only the top 5 predictions
        top_preds = preds[:5]

        if target in top_preds:
            # Rank is 0-indexed in list, so add 1.
            # If target is at index 0, rank is 1, score is 1/1.
            # If target is at index 4, rank is 5, score is 1/5.
            rank = top_preds.index(target) + 1
            total_score += 1.0 / rank

    return total_score / n_samples
