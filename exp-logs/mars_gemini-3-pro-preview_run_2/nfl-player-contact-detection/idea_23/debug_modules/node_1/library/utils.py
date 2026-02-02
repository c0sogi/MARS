import os
import random
import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef
from library.config import Config


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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def optimize_threshold(y_true, y_pred_probs):
    """
    Finds the best decision threshold that maximizes the Matthews Correlation Coefficient (MCC).

    Args:
        y_true (np.array or torch.Tensor): Ground truth binary labels.
        y_pred_probs (np.array or torch.Tensor): Predicted probabilities (logits or sigmoid output).
                                                 If logits, ensure they are converted to probabilities
                                                 before passing or handle accordingly.
                                                 This function assumes probabilities (0-1).

    Returns:
        float: The optimal threshold.
        float: The best MCC score achieved.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred_probs, torch.Tensor):
        y_pred_probs = y_pred_probs.detach().cpu().numpy()

    # Flatten arrays to ensure 1D
    y_true = y_true.ravel()
    y_pred_probs = y_pred_probs.ravel()

    best_mcc = -1.0
    best_threshold = 0.5

    # Generate thresholds based on Config
    thresholds = np.linspace(
        Config.THRESHOLD_RANGE[0], Config.THRESHOLD_RANGE[1], Config.THRESHOLD_STEPS
    )

    for thresh in thresholds:
        # Binarize predictions
        y_pred_bin = (y_pred_probs >= thresh).astype(int)

        # Calculate MCC
        current_mcc = matthews_corrcoef(y_true, y_pred_bin)

        if current_mcc > best_mcc:
            best_mcc = current_mcc
            best_threshold = thresh

    return best_threshold, best_mcc
