import os
import random
import copy
import numpy as np
import torch
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ModelEMA:
    """
    Exponential Moving Average (EMA) for model weights.
    Maintains a shadow copy of the model that updates slowly during training.
    This often leads to better generalization and stability.
    """

    def __init__(self, model, decay=0.999, device=None):
        """
        Args:
            model (nn.Module): The model to track.
            decay (float): The decay factor for EMA (default: 0.999).
            device (torch.device): The device to store the EMA model on.
        """
        self.decay = decay
        # Create a deep copy of the model for EMA
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()

        if device:
            self.ema_model.to(device)

        # Ensure EMA model parameters do not require gradients
        for param in self.ema_model.parameters():
            param.requires_grad = False

    def update(self, model):
        """
        Update the EMA model parameters based on the current model state.

        Args:
            model (nn.Module): The current training model.
        """
        with torch.no_grad():
            # Update parameters: ema_param = decay * ema_param + (1 - decay) * current_param
            for ema_param, param in zip(
                self.ema_model.parameters(), model.parameters()
            ):
                ema_param.mul_(self.decay).add_(param.data, alpha=1 - self.decay)

            # Update buffers (e.g., BatchNorm running stats) by copying them directly
            for ema_buf, buf in zip(self.ema_model.buffers(), model.buffers()):
                ema_buf.copy_(buf)


def map_fine_to_coarse(predictions):
    """
    Maps fine-grained predictions (30+ classes) to the 12 competition target labels.

    Args:
        predictions: A single string label or a list/array of string labels.

    Returns:
        The mapped label(s) as a string or a list of strings.
    """
    if isinstance(predictions, (list, np.ndarray)):
        return [Config.map_fine_grained_to_target(label) for label in predictions]
    else:
        return Config.map_fine_grained_to_target(predictions)
