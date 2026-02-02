import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

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

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_map5(predictions, targets):
    """
    Calculates the Mean Average Precision @ 5 (MAP@5) for single-label classification.

    Args:
        predictions (list of list): A list where each element is a list of the top 5 predicted labels.
        targets (list): A list of the ground truth labels (one per sample).

    Returns:
        float: The MAP@5 score.
    """
    if len(predictions) != len(targets):
        raise ValueError(
            f"Input length mismatch: predictions ({len(predictions)}) vs targets ({len(targets)})"
        )

    n_samples = len(targets)
    if n_samples == 0:
        return 0.0

    total_score = 0.0

    for preds, target in zip(predictions, targets):
        # Consider only the top 5 predictions
        top_preds = preds[:5]

        # Calculate AP for this sample
        # Since there is exactly one correct label, AP is 1/rank if found, else 0.
        ap = 0.0
        for rank, pred_label in enumerate(top_preds):
            if pred_label == target:
                # rank is 0-indexed, so we use rank + 1
                ap = 1.0 / (rank + 1)
                break

        total_score += ap

    return total_score / n_samples
