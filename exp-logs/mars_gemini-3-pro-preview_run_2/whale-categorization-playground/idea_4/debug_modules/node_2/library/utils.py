import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def map_at_5(predictions, ground_truth):
    """
    Calculates the Mean Average Precision at 5 (MAP@5).

    The metric is calculated as the mean of the Average Precision (AP) for each sample.
    Since there is only one correct label per image, the AP for a sample is 1/(rank+1)
    if the correct label is in the top 5 predictions, and 0 otherwise.

    Args:
        predictions (list of list): A list where each element is a list of predicted labels (strings).
                                    Only the first 5 elements of each inner list are considered.
        ground_truth (list): A list of actual labels (strings).

    Returns:
        float: The MAP@5 score.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"Size mismatch: predictions ({len(predictions)}) vs ground_truth ({len(ground_truth)})"
        )

    total_score = 0.0
    n_samples = len(predictions)

    if n_samples == 0:
        return 0.0

    for preds, truth in zip(predictions, ground_truth):
        # Consider only the top 5 predictions
        top_5_preds = preds[:5]

        if truth in top_5_preds:
            # List index is 0-based, rank is 1-based
            rank = top_5_preds.index(truth)
            total_score += 1.0 / (rank + 1)

    return total_score / n_samples
