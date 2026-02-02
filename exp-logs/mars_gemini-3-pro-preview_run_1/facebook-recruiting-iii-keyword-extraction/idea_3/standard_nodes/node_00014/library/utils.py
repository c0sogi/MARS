import os
import random
import numpy as np
import torch
from sklearn.metrics import f1_score
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_f1_score(outputs, targets, threshold=0.5):
    """
    Computes the Mean F1-Score (samples average) for multi-label classification.

    Args:
        outputs (torch.Tensor or np.ndarray): Model predictions (logits or probabilities).
        targets (torch.Tensor or np.ndarray): Ground truth binary labels.
        threshold (float): Threshold to convert probabilities to binary predictions.

    Returns:
        float: The mean F1 score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Check if outputs are logits (heuristically) and apply sigmoid if so
    # If values are outside [0, 1], they are likely logits.
    if outputs.min() < 0 or outputs.max() > 1:
        # Apply sigmoid: 1 / (1 + exp(-x))
        outputs = 1 / (1 + np.exp(-outputs))

    # Binarize predictions
    preds = (outputs > threshold).astype(int)
    targets = targets.astype(int)

    # Calculate F1 score with 'samples' average
    # 'samples': Calculate metrics for each instance, and find their average
    return f1_score(targets, preds, average="samples", zero_division=0)
