import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import CFG


def seed_everything(seed: int = 42):
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_score(y_true, y_pred):
    """
    Calculates the Mean Column-wise ROC AUC.

    Args:
        y_true (np.array): Ground truth labels (One-hot encoded or Multilabel binary indicators).
                           Shape: (N_samples, N_classes)
        y_pred (np.array): Predicted probabilities.
                           Shape: (N_samples, N_classes)

    Returns:
        float: The mean column-wise ROC AUC score.
    """
    # sklearn's roc_auc_score with average='macro' computes the metric for each label,
    # and finds their unweighted mean. This matches "Mean column-wise ROC AUC".
    try:
        return roc_auc_score(y_true, y_pred, average="macro")
    except ValueError:
        # Handle cases where a specific class might not be present in the batch/fold
        # by falling back to a safe calculation or returning 0.5 if completely undefined
        return 0.5


def calculate_class_weights(df, target_col=None, class_labels=None):
    """
    Calculates inverse frequency weights for class balancing.

    Args:
        df (pd.DataFrame): The training dataframe containing the target labels.
        target_col (str, optional): The column name containing the class labels.
                                    Defaults to CFG.target_col.
        class_labels (list, optional): List of class names in the specific order
                                       corresponding to the model's output indices.
                                       Defaults to CFG.class_labels.

    Returns:
        torch.Tensor: A tensor of weights for each class.
    """
    if target_col is None:
        target_col = CFG.target_col

    if class_labels is None:
        class_labels = CFG.class_labels

    # Calculate counts for each class
    class_counts = df[target_col].value_counts().to_dict()

    # Ensure all classes are present in counts, default to 0 (though should not happen in train set)
    counts = [class_counts.get(l, 0) for l in class_labels]

    # Total samples
    total_samples = sum(counts)
    num_classes = len(class_labels)

    # Compute weights: w_j = N / (C * n_j)
    # Adding a small epsilon to avoid division by zero if a class is missing in a subset
    weights = [total_samples / (num_classes * (c + 1e-6)) for c in counts]

    return torch.tensor(weights, dtype=torch.float32)
