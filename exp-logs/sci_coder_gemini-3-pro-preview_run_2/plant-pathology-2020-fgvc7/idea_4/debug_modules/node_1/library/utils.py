import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Seeds all random number generators to ensure reproducibility.

    Args:
        seed (int): The random seed to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    """
    Worker initialization function for PyTorch DataLoader to ensure
    deterministic data loading and augmentation.

    Args:
        worker_id (int): The ID of the worker process.
    """
    # Use the initial seed from the main process, shifted by worker_id
    # Modulo 2**32 to ensure it fits in a 32-bit integer for numpy
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def calculate_metric(y_true, y_pred):
    """
    Calculates the Mean Column-wise ROC AUC.

    Args:
        y_true (np.ndarray): Ground truth labels (N, num_classes), one-hot encoded.
        y_pred (np.ndarray): Predicted probabilities (N, num_classes).

    Returns:
        float: The mean column-wise ROC AUC score.
    """
    try:
        # Calculate macro-average ROC AUC (mean of column-wise scores)
        return roc_auc_score(y_true, y_pred, average="macro")
    except ValueError:
        # Fallback for edge cases (e.g., a class is missing in the current batch)
        scores = []
        num_classes = y_true.shape[1]
        for i in range(num_classes):
            try:
                score = roc_auc_score(y_true[:, i], y_pred[:, i])
                scores.append(score)
            except ValueError:
                # If a class has only one unique value in y_true (e.g., all 0s),
                # ROC AUC is undefined. We assign 0.5 (random guessing).
                scores.append(0.5)
        return np.mean(scores)
