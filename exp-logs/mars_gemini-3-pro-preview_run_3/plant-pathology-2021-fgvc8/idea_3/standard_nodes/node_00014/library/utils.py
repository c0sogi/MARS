import os
import random
import numpy as np
import torch
from sklearn.metrics import f1_score


def seed_everything(seed: int):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Enforce deterministic algorithms for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """
    Returns the appropriate PyTorch device (CUDA if available, else CPU).

    Returns:
        torch.device: The selected device.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_f1_score(logits, targets, threshold=0.5):
    """
    Calculates the Mean F1-Score (Macro-averaged) for multi-label classification.

    Args:
        logits (torch.Tensor): Raw output from the model (before sigmoid).
        targets (torch.Tensor): Ground truth labels (multi-hot encoded).
        threshold (float): Threshold for converting probabilities to binary predictions.

    Returns:
        float: The macro-averaged F1 score.
    """
    # Ensure inputs are tensors
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits)
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets)

    # Detach and move to CPU for numpy conversion
    logits = logits.detach().cpu()
    targets = targets.detach().cpu()

    # Apply sigmoid activation to get probabilities
    probs = torch.sigmoid(logits)

    # Convert probabilities to binary predictions based on threshold
    preds = (probs > threshold).int().numpy()
    targets = targets.int().numpy()

    # Calculate Macro F1 Score
    # 'macro' calculates metrics for each label, and finds their unweighted mean.
    # zero_division=0 ensures that if a class has no positive predictions, the score is 0 instead of erroring.
    return f1_score(targets, preds, average="macro", zero_division=0)
