import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed: int = Config.seed) -> None:
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """
    Returns the PyTorch device configured in Config.

    Returns:
        torch.device: The device object (cpu or cuda).
    """
    return Config.device


def calculate_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculates the multiclass accuracy.

    Args:
        output (torch.Tensor): Raw model outputs (logits) of shape (batch_size, num_classes).
        target (torch.Tensor): Ground truth labels of shape (batch_size,).

    Returns:
        float: The accuracy (0.0 to 1.0).
    """
    with torch.no_grad():
        # Get the index of the max log-probability
        _, predicted = torch.max(output, dim=1)
        correct = (predicted == target).sum().item()
        total = target.size(0)

        if total == 0:
            return 0.0

        return correct / total


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and accuracy during training epochs.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(
    state: dict, is_best: bool, filepath: str = Config.best_model_path
) -> None:
    """
    Saves the model checkpoint.

    Args:
        state (dict): The state dictionary containing model parameters, optimizer state, etc.
        is_best (bool): If True, saves/overwrites the checkpoint at filepath.
        filepath (str): Path to save the checkpoint. Defaults to Config.best_model_path.
    """
    if is_best:
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(state, filepath)


def count_parameters(model: torch.nn.Module) -> int:
    """
    Counts the number of trainable parameters in a model.

    Args:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
