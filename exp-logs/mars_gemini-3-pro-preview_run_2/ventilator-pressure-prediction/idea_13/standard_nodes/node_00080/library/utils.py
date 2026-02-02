import os
import random
import numpy as np
import torch
import torch.nn as nn
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Enforces deterministic behavior in cuDNN.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Enforce deterministic behavior for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class WeightedL1Loss(nn.Module):
    """
    Weighted L1 Loss function.
    Assigns different weights to the inspiratory and expiratory phases of the breath.

    Weights:
        Inspiratory (u_out=0): Config.INSPIRATORY_WEIGHT (1.0)
        Expiratory (u_out=1): Config.EXPIRATORY_WEIGHT (0.1)
    """

    def __init__(self):
        super(WeightedL1Loss, self).__init__()
        self.insp_weight = Config.INSPIRATORY_WEIGHT
        self.exp_weight = Config.EXPIRATORY_WEIGHT

    def forward(self, preds, targets, u_out):
        """
        Calculates the weighted mean absolute error.

        Args:
            preds (torch.Tensor): Predicted pressure values.
            targets (torch.Tensor): Actual pressure values.
            u_out (torch.Tensor): Control input 'u_out' (0 for inspiratory, 1 for expiratory).

        Returns:
            torch.Tensor: The scalar weighted loss.
        """
        # Calculate absolute error
        abs_error = torch.abs(preds - targets)

        # Determine weights based on u_out
        # If u_out is 0 (Inspiratory), weight is insp_weight
        # If u_out is 1 (Expiratory), weight is exp_weight
        weights = (1 - u_out) * self.insp_weight + u_out * self.exp_weight

        # Apply weights
        weighted_error = abs_error * weights

        # Return mean loss
        return weighted_error.mean()


def save_checkpoint(state, is_best, filename="checkpoint.pth"):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model_state_dict, optimizer_state_dict, etc.
        is_best (bool): If True, copies this checkpoint to the BEST_MODEL_PATH.
        filename (str): Name of the checkpoint file to save in WORKING_DIR.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    filepath = os.path.join(Config.WORKING_DIR, filename)
    torch.save(state, filepath)

    if is_best:
        torch.save(state, Config.BEST_MODEL_PATH)


def load_checkpoint(model, optimizer=None, scheduler=None, path=Config.BEST_MODEL_PATH):
    """
    Loads a model checkpoint.

    Args:
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        scheduler (torch.optim.lr_scheduler, optional): The scheduler to load state into.
        path (str): Path to the checkpoint file.

    Returns:
        dict: The loaded checkpoint dictionary, or None if file not found.
    """
    if not os.path.exists(path):
        return None

    # Load checkpoint to the configured device
    checkpoint = torch.load(path, map_location=Config.DEVICE)

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer state if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state if provided
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
