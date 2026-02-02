import os
import random
import numpy as np
import torch
from sklearn.metrics import log_loss
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for various libraries to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clip_probabilities(probs):
    """
    Clips probabilities to the range [1e-15, 1 - 1e-15] to avoid log(0) and
    adhere to the competition metric specification.
    """
    return np.clip(probs, 1e-15, 1 - 1e-15)


def multiclass_log_loss(y_true, y_pred):
    """
    Computes the multi-class logarithmic loss after normalizing and clipping probabilities.

    Args:
        y_true: Array-like of shape (n_samples,) containing true class indices or labels.
        y_pred: Array-like of shape (n_samples, n_classes) containing predicted probabilities.

    Returns:
        float: The calculated log loss.
    """
    # Ensure numpy array for processing
    y_pred = np.array(y_pred)

    # 1. Rescale: each row is divided by the row sum (as per task description)
    row_sums = y_pred.sum(axis=1)
    # Handle potential zero sums to avoid division by zero (though unlikely with softmax)
    row_sums[row_sums == 0] = 1.0
    y_pred = y_pred / row_sums[:, np.newaxis]

    # 2. Clip probabilities
    y_pred = clip_probabilities(y_pred)

    # 3. Compute Log Loss
    # We provide labels to ensure correct mapping even if a batch misses a class
    labels = list(range(Config.NUM_CLASSES))
    return log_loss(y_true, y_pred, labels=labels)
