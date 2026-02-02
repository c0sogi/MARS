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
        # Deterministic mode can be slower but is required for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value of a metric.
    Useful for tracking loss and accuracy during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Update the meter with a new value.

        Args:
            val (float): The current value to add.
            n (int): The number of samples this value represents (usually batch size).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def map_per_image(label, predictions):
    """
    Computes the Average Precision (AP) for a single image.

    The score is 1/(k+1) if the correct label is at index k (0-indexed)
    in the predictions, and 0 otherwise. Only the top 5 predictions are considered.

    Args:
        label (str or int): The ground truth label.
        predictions (list): Ordered list of predicted labels.

    Returns:
        float: The AP score for this image.
    """
    try:
        # We only care about the top 5 predictions
        return 1.0 / (predictions[:5].index(label) + 1.0)
    except ValueError:
        return 0.0


def map5(labels, predictions):
    """
    Computes the Mean Average Precision at 5 (MAP@5) across a dataset.

    Args:
        labels (list or np.array): List of ground truth labels.
        predictions (list of lists): List where each element is a list of predicted labels.

    Returns:
        float: The MAP@5 score.
    """
    if len(labels) != len(predictions):
        raise ValueError(
            f"Length mismatch: labels ({len(labels)}) vs predictions ({len(predictions)})"
        )

    score = 0.0
    for label, preds in zip(labels, predictions):
        score += map_per_image(label, preds)

    return score / len(labels)
