import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # deterministic algorithms for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state: dict, filename: str = Config.MODEL_CHECKPOINT):
    """
    Saves the model checkpoint to the specified file.

    Args:
        state (dict): The state dictionary containing model parameters, optimizer state, etc.
        filename (str): The path where the checkpoint will be saved. Defaults to Config.MODEL_CHECKPOINT.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    filename: str = Config.MODEL_CHECKPOINT,
):
    """
    Loads the model checkpoint from the specified file.

    Args:
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        filename (str): The path to the checkpoint file. Defaults to Config.MODEL_CHECKPOINT.

    Returns:
        dict: The loaded checkpoint dictionary.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint file not found: {filename}")

    # Load onto the configured device
    checkpoint = torch.load(filename, map_location=Config.DEVICE)

    model.load_state_dict(checkpoint["state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training.
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
