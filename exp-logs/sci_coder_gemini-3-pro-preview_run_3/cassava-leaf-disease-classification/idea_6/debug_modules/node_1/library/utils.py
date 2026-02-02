import os
import random
import numpy as np
import torch


def seed_everything(seed: int):
    """
    Seeds all random number generators to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_accuracy(output, target):
    """
    Computes the classification accuracy.

    Args:
        output: Predicted logits, probabilities (N, C), or class indices (N,).
                Can be torch.Tensor or numpy.ndarray.
        target: Ground truth labels (N,).
                Can be torch.Tensor or numpy.ndarray.

    Returns:
        float: The accuracy score.
    """
    # Convert torch Tensors to numpy arrays if necessary
    if isinstance(output, torch.Tensor):
        output = output.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    # If output is 2D (logits or probabilities), take the argmax to get class indices
    if output.ndim > 1:
        preds = np.argmax(output, axis=1)
    else:
        preds = output

    # Flatten arrays to ensure 1D comparison
    preds = preds.flatten()
    target = target.flatten()

    # Calculate accuracy
    return np.mean(preds == target)
