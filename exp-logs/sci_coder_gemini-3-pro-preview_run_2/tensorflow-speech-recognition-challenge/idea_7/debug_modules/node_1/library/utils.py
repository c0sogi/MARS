import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    # Python random
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU if applicable

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Python Hash Seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def calculate_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Computes the multiclass accuracy for a batch of predictions.

    Args:
        outputs (torch.Tensor): Model predictions (logits or probabilities) of shape (batch_size, num_classes).
        targets (torch.Tensor): Ground truth labels of shape (batch_size).

    Returns:
        float: The accuracy score (0.0 to 1.0).
    """
    with torch.no_grad():
        # Get the predicted class index by finding the max value along the class dimension
        _, preds = torch.max(outputs, dim=1)

        # Calculate the number of correct predictions
        correct = (preds == targets).sum().item()

        # Calculate total number of samples
        total = targets.size(0)

        # Avoid division by zero
        if total == 0:
            return 0.0

        return correct / total
