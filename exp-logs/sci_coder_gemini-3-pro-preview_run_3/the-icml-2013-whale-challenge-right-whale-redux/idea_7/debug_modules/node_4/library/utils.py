import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import TrainConfig


def set_seed(seed=TrainConfig.seed):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to TrainConfig.seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Computes the Mixup loss by mixing the weighted scalar losses of the input pair.

    This implementation mixes the loss values rather than the labels. This ensures that
    if the criterion includes class weighting (e.g., pos_weight in BCEWithLogitsLoss),
    the gradients from the minority class are not diluted by the label mixing process.

    Args:
        criterion (callable): The loss function. It should accept (pred, target) and return a scalar loss.
                              Ideally, this criterion already incorporates class weights (e.g., pos_weight).
        pred (torch.Tensor): The predictions from the model.
        y_a (torch.Tensor): The target labels for the first image in the mix.
        y_b (torch.Tensor): The target labels for the second image in the mix.
        lam (float): The mixup lambda coefficient (weight for y_a).

    Returns:
        torch.Tensor: The mixed loss value.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def compute_auc(y_true, y_pred):
    """
    Computes the Area Under the ROC Curve (AUC).

    Args:
        y_true (array-like): Ground truth binary labels.
        y_pred (array-like): Predicted probabilities.

    Returns:
        float: The AUC score. Returns 0.5 if only one class is present in y_true.
    """
    # Detach from graph if tensors and convert to numpy
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # Check for single class edge case to avoid sklearn error
    if len(np.unique(y_true)) < 2:
        return 0.5

    return roc_auc_score(y_true, y_pred)


def calculate_pos_weight(labels):
    """
    Calculates the positive class weight for handling class imbalance.
    Formula: weight = number_of_negatives / number_of_positives

    Args:
        labels (array-like): List or array of binary labels (0/1).

    Returns:
        torch.Tensor: The calculated weight as a float tensor.
    """
    labels = np.array(labels)
    num_pos = np.sum(labels == 1)
    num_neg = np.sum(labels == 0)

    # Avoid division by zero
    if num_pos == 0:
        return torch.tensor(1.0, dtype=torch.float32)

    weight = num_neg / num_pos
    return torch.tensor(weight, dtype=torch.float32)
