import os
import random
import shutil
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.seed):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to Config.seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def save_checkpoint(state, is_best, filename):
    """
    Saves the model state to a file.

    Args:
        state (dict): The state dictionary containing model weights, optimizer state, etc.
        is_best (bool): Flag indicating if this is the best model so far.
        filename (str): The path where the checkpoint should be saved.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save the checkpoint
    torch.save(state, filename)

    # If it is the best model, create a copy named 'best_model.pth'
    if is_best:
        dirname = os.path.dirname(filename)
        best_path = os.path.join(dirname, "best_model.pth")
        shutil.copyfile(filename, best_path)


def load_checkpoint(
    model, filename, device=Config.device, optimizer=None, scheduler=None
):
    """
    Loads a checkpoint into the model, and optionally into the optimizer and scheduler.

    Args:
        model (torch.nn.Module): The model to load weights into.
        filename (str): Path to the checkpoint file.
        device (str): Device to map the location to. Defaults to Config.device.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        scheduler (object, optional): Scheduler to load state into.

    Returns:
        dict: The loaded checkpoint dictionary (useful for retrieving epoch or best_score), or None if not found.
    """
    if not os.path.isfile(filename):
        print(f"No checkpoint found at '{filename}'")
        return None

    print(f"Loading checkpoint '{filename}'")
    checkpoint = torch.load(filename, map_location=device)

    # Extract state_dict, handling cases where it might be nested or direct
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Handle DataParallel wrapping ('module.' prefix) mismatch
    # If the checkpoint has 'module.' prefix but the current model does not
    if list(state_dict.keys())[0].startswith("module.") and not hasattr(
        model, "module"
    ):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)

    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint
