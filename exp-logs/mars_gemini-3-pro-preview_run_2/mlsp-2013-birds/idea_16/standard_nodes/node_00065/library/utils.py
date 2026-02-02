import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


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

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Returns the appropriate PyTorch device (CUDA if available, else CPU).

    Returns:
        torch.device: The device object.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def calculate_metrics(y_true, y_pred):
    """
    Computes the Area Under the ROC Curve (AUC) for multi-label classification.

    Args:
        y_true (np.ndarray): Ground truth binary labels of shape (n_samples, n_classes).
        y_pred (np.ndarray): Predicted probabilities of shape (n_samples, n_classes).

    Returns:
        float: The macro-averaged ROC AUC score.
    """
    # Ensure inputs are numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    try:
        # Compute Macro-Average ROC AUC
        # 'macro': Calculate metrics for each label, and find their unweighted mean.
        # This does not take label imbalance into account.
        score = roc_auc_score(y_true, y_pred, average="macro")
    except ValueError as e:
        # This can happen if a class has no positive samples in the validation set
        # In such cases, we can try 'micro' or handle it gracefully.
        # For this competition, we return 0.5 or print a warning if strictly needed,
        # but usually validation sets are stratified to avoid this.
        print(f"Warning during metric calculation: {e}")
        score = 0.5

    return score
