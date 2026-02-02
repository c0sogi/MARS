import os
import random
import numpy as np
import torch
from copy import deepcopy
from sklearn.metrics import cohen_kappa_score


def seed_everything(seed: int = 42):
    """
    Sets the seed for generating random numbers to ensure reproducibility.

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


def quadratic_weighted_kappa(y_true, y_pred):
    """
    Calculates the quadratic weighted kappa (QWK) score.

    This metric measures the agreement between two ratings. This metric typically
    varies from 0 (random agreement) to 1 (complete agreement). It can be negative
    if there is less agreement than expected by chance.

    Args:
        y_true: Array-like of ground truth labels (integers 0-4).
        y_pred: Array-like of predicted labels (integers 0-4).

    Returns:
        float: The quadratic weighted kappa score.
    """
    # Handle PyTorch Tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are integer arrays (QWK is defined for discrete ratings)
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


class ModelEMA:
    """
    Model Exponential Moving Average (EMA).

    Maintains a moving average of model parameters and buffers.
    Using EMA weights often leads to better generalization and stability
    compared to using the final trained weights.
    """

    def __init__(self, model, decay=0.999, device=None):
        """
        Args:
            model (torch.nn.Module): The model to track.
            decay (float): The decay factor for the moving average (beta).
                           Value close to 1.0 (e.g., 0.999, 0.9999) gives more smoothing.
            device (torch.device, optional): Device to store the EMA model on.
        """
        self.decay = decay
        # Create a deep copy of the model to serve as the shadow model
        self.ema = deepcopy(model)
        self.ema.eval()

        if device:
            self.ema.to(device)

        # Disable gradients for the shadow model to save memory/compute
        for param in self.ema.parameters():
            param.requires_grad = False

    def update(self, model):
        """
        Update the shadow model parameters using the current model parameters.

        Formula:
            shadow_variable = decay * shadow_variable + (1 - decay) * current_variable

        Args:
            model (torch.nn.Module): The current training model.
        """
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema.state_dict()

            for k, v in esd.items():
                if k in msd:
                    model_v = msd[k].detach()
                    # Ensure device/dtype compatibility
                    if model_v.device != v.device:
                        model_v = model_v.to(v.device)
                    if model_v.dtype != v.dtype:
                        model_v = model_v.to(dtype=v.dtype)

                    # Update shadow weight
                    v.copy_(self.decay * v + (1.0 - self.decay) * model_v)

    def set(self, model):
        """
        Forcefully set the shadow model weights to match the current model.
        Useful for initialization or resetting.
        """
        self.ema.load_state_dict(model.state_dict())
