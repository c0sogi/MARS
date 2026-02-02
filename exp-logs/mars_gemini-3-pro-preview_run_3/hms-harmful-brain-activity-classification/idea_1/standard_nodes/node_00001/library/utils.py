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
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(
    state: dict, is_best: bool = False, filename: str = "checkpoint.pth"
):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model_state_dict, optimizer_state_dict, etc.
        is_best (bool): If True, saves a copy as 'best_model.pth'.
        filename (str): The name of the checkpoint file.
    """
    # Ensure the working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    filepath = os.path.join(Config.WORKING_DIR, filename)
    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
        torch.save(state, best_path)


def load_checkpoint(
    model, optimizer=None, filename: str = "best_model.pth", device: str = Config.DEVICE
):
    """
    Loads a model checkpoint.

    Args:
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        filename (str): The filename of the checkpoint to load.
        device (str): The device to map the checkpoint to.

    Returns:
        dict: The full checkpoint dictionary (useful for retrieving epoch, loss, etc.).
        None: If the file does not exist.
    """
    filepath = os.path.join(Config.WORKING_DIR, filename)

    if not os.path.exists(filepath):
        print(f"Checkpoint file not found at {filepath}")
        return None

    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training loops.
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


def print_metrics(metrics: dict, prefix: str = ""):
    """
    Prints metric values with full precision.

    Args:
        metrics (dict): Dictionary of metric names and values.
        prefix (str): Optional prefix string (e.g., "Validation").
    """
    msg_parts = []
    if prefix:
        msg_parts.append(f"[{prefix}]")

    for k, v in metrics.items():
        # Print full precision without formatting
        msg_parts.append(f"{k}: {v}")

    print(" ".join(msg_parts))
