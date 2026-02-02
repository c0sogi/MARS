import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def map_at_5(predictions, ground_truth):
    """
    Calculates the Mean Average Precision at 5 (MAP@5).

    The metric is defined as the mean of the Average Precision (AP) for each query.
    Since there is only one correct label per image, the AP for a query is simply
    1/rank if the correct label is in the top 5 predictions (where rank is 1-based),
    and 0 otherwise.

    Args:
        predictions (list of list of str): A list where each element is a list of
                                           predicted class labels (strings), ordered by confidence.
                                           Only the top 5 predictions are considered.
        ground_truth (list of str): A list of the true class labels (strings) for each query.

    Returns:
        float: The MAP@5 score.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"Size mismatch: predictions ({len(predictions)}) vs ground_truth ({len(ground_truth)})"
        )

    total_score = 0.0
    num_samples = len(ground_truth)

    if num_samples == 0:
        return 0.0

    for preds, true_label in zip(predictions, ground_truth):
        # Ensure we only look at the top 5 predictions
        top_preds = preds[:5]

        if true_label in top_preds:
            # list.index returns 0-based index. Rank is index + 1.
            rank = top_preds.index(true_label) + 1
            total_score += 1.0 / rank

    return total_score / num_samples
