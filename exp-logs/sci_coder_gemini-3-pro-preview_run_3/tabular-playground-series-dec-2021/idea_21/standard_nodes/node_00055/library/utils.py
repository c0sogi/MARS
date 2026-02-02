import os
import random
import numpy as np
import torch
from library.config import SEED


def seed_everything(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to SEED from config.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accuracy_score(outputs, targets):
    """
    Computes the accuracy of predictions against targets.

    Args:
        outputs (torch.Tensor or np.ndarray): Model outputs (logits of shape [N, C])
                                              or predicted classes (shape [N]).
        targets (torch.Tensor or np.ndarray): Ground truth labels (shape [N]).

    Returns:
        float: The accuracy score (0.0 to 1.0).
    """
    # Convert numpy arrays to tensors if necessary
    if isinstance(outputs, np.ndarray):
        outputs = torch.from_numpy(outputs)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)

    # Detach and move to CPU for calculation
    outputs = outputs.detach().cpu()
    targets = targets.detach().cpu()

    # If outputs are logits (2D), take argmax to get class indices
    if outputs.dim() > 1:
        _, predictions = torch.max(outputs, dim=1)
    else:
        predictions = outputs

    correct = (predictions == targets).sum().item()
    total = targets.size(0)

    if total == 0:
        return 0.0

    return correct / total


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training steps.
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


def get_device():
    """Returns the appropriate torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
