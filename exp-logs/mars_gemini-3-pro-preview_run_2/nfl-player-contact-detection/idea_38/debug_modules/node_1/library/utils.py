import os
import random
import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef
from library.config import SEED


def seed_everything(seed: int = SEED):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to use. Defaults to SEED from config.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU, though we have 1
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_mcc(y_true, y_logits, threshold: float = 0.5) -> float:
    """
    Computes the Matthews Correlation Coefficient (MCC) given logits and labels.

    Args:
        y_true: Ground truth binary labels (numpy array or torch Tensor).
        y_logits: Raw model output logits (numpy array or torch Tensor).
        threshold: Probability threshold to determine class 1.

    Returns:
        float: The MCC score.
    """
    # Convert torch tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_logits, torch.Tensor):
        y_logits = y_logits.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_logits = np.asarray(y_logits)

    # Flatten arrays to ensure 1D
    y_true = y_true.ravel()
    y_logits = y_logits.ravel()

    # Apply Sigmoid to convert logits to probabilities in a numerically stable way
    # sigmoid(x) = 1 / (1 + exp(-x))
    # Stable version:
    # For x >= 0: 1 / (1 + exp(-x))
    # For x < 0:  exp(x) / (1 + exp(x))
    y_probs = np.where(
        y_logits >= 0,
        1.0 / (1.0 + np.exp(-y_logits)),
        np.exp(y_logits) / (1.0 + np.exp(y_logits)),
    )

    # Binarize predictions based on threshold
    y_pred = (y_probs >= threshold).astype(int)

    # Calculate MCC
    mcc = matthews_corrcoef(y_true, y_pred)

    return float(mcc)
