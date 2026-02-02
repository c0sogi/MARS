import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_map5(predictions, targets):
    """
    Calculates the Mean Average Precision @ 5 (MAP@5).

    Args:
        predictions (list or np.array): A list of lists (or 2D array), where each inner sequence
                                        contains the top predicted labels (strings or ints).
                                        Only the first 5 predictions per sample are considered.
        targets (list or np.array): A list or 1D array of ground truth labels (strings or ints).

    Returns:
        float: The MAP@5 score.
    """
    # Convert numpy arrays to lists for consistent indexing and handling
    if isinstance(predictions, np.ndarray):
        predictions = predictions.tolist()
    if isinstance(targets, np.ndarray):
        targets = targets.tolist()

    # Basic validation
    if len(predictions) != len(targets):
        raise ValueError(
            f"Size mismatch: predictions ({len(predictions)}) vs targets ({len(targets)})"
        )

    scores = []

    for i, pred_row in enumerate(predictions):
        target = targets[i]

        # Take top 5 predictions
        top_preds = pred_row[:5]

        if target in top_preds:
            # Rank is 0-based index
            rank = top_preds.index(target)
            # Score is 1 / (rank + 1)
            scores.append(1.0 / (rank + 1))
        else:
            scores.append(0.0)

    if not scores:
        return 0.0

    return float(np.mean(scores))
