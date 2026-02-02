import os
import random
import numpy as np
import torch
from library.config import TRAIN_CONFIG


def set_seed(seed=TRAIN_CONFIG["seed"]):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set. Defaults to the seed in TRAIN_CONFIG.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_accuracy(outputs, targets):
    """
    Computes the multiclass accuracy for a batch of predictions.

    Args:
        outputs (torch.Tensor): Logits or probabilities from the model of shape (Batch, NumClasses).
        targets (torch.Tensor): Ground truth labels of shape (Batch).

    Returns:
        float: The accuracy (ratio of correct predictions) as a float.
    """
    with torch.no_grad():
        # Get the index of the max log-probability along the class dimension
        _, predicted = torch.max(outputs, 1)

        # Calculate the number of correct predictions
        correct = (predicted == targets).sum().item()

        # Calculate accuracy
        accuracy = correct / targets.size(0)

    return accuracy
