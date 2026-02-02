import os
import random
import numpy as np
import torch
from copy import deepcopy
from sklearn.metrics import cohen_kappa_score
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_kappa_score(y_true, y_pred):
    """
    Computes the Quadratic Weighted Kappa (QWK) score, measuring agreement between
    two raters (ground truth vs prediction).

    Args:
        y_true (array-like): Ground truth labels (integers 0-4).
        y_pred (array-like): Predicted labels (integers 0-4).

    Returns:
        float: The Quadratic Weighted Kappa score.
    """
    # Ensure inputs are numpy arrays of integers
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    # Calculate score using sklearn's implementation with quadratic weights
    score = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return score


class ModelEMA:
    """
    Model Exponential Moving Average (EMA).
    Maintains a moving average of model parameters to improve model robustness
    and generalization, particularly useful for noisy gradients or complex loss landscapes.
    """

    def __init__(self, model, decay=0.999, device=None):
        """
        Initialize the EMA model.

        Args:
            model (torch.nn.Module): The source model to track.
            decay (float): The decay factor for the moving average (default: 0.999).
            device (torch.device, optional): The device to store the EMA model on.
                                           If None, defaults to the source model's device.
        """
        self.decay = decay
        # Create a deep copy of the model to serve as the shadow model
        self.module = deepcopy(model)
        self.module.eval()

        # Move to specified device if provided
        if device is not None:
            self.module.to(device)

        # Disable gradients for the EMA model as it is updated via averaging, not backprop
        for param in self.module.parameters():
            param.requires_grad = False

    def update(self, model):
        """
        Update the EMA model parameters using the current state of the source model.

        Args:
            model (torch.nn.Module): The current training model.
        """
        with torch.no_grad():
            # Update learnable parameters using the moving average formula:
            # ema_param = decay * ema_param + (1 - decay) * current_param
            for ema_v, model_v in zip(self.module.parameters(), model.parameters()):
                # Ensure tensors are on the same device before operation
                if ema_v.device != model_v.device:
                    model_v = model_v.to(ema_v.device)

                ema_v.copy_(self.decay * ema_v + (1.0 - self.decay) * model_v)

            # Update buffers (e.g., Batch Norm running statistics)
            # Buffers are typically copied directly rather than averaged to maintain
            # accurate statistics of the current data distribution.
            for ema_v, model_v in zip(self.module.buffers(), model.buffers()):
                if ema_v.device != model_v.device:
                    model_v = model_v.to(ema_v.device)

                ema_v.copy_(model_v)
