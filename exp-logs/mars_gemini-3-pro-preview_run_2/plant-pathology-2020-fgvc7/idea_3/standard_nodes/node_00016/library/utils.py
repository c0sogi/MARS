import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_worker_init_fn(seed=42):
    """
    Returns a worker initialization function for PyTorch DataLoaders to ensure
    deterministic data loading.

    Args:
        seed (int): The base seed value.

    Returns:
        function: A function that takes `worker_id` as input.
    """

    def worker_init_fn(worker_id):
        # Create a unique seed for each worker based on the global seed
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return worker_init_fn


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the mean column-wise ROC AUC score.

    Args:
        y_true (np.ndarray): Ground truth labels (one-hot encoded or multi-label binary),
                             shape (n_samples, n_classes).
        y_pred (np.ndarray): Predicted probabilities, shape (n_samples, n_classes).

    Returns:
        float: The mean column-wise ROC AUC.
    """
    # average='macro' calculates metrics for each label, and finds their unweighted mean.
    # multi_class='ovr' (One-vs-Rest) is appropriate for multi-label or one-hot encoded multi-class.
    return roc_auc_score(y_true, y_pred, average="macro", multi_class="ovr")
