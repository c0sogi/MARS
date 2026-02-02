import os
import random
import copy
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for various libraries to ensure reproducibility.

    Args:
        seed (int): The random seed value. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ModelEMA:
    """
    Maintains an Exponential Moving Average (EMA) of the model weights.
    This helps in stabilizing training and often leads to better generalization.
    """

    def __init__(self, model, decay=Config.EMA_DECAY, device=None):
        """
        Args:
            model (torch.nn.Module): The model to track.
            decay (float): The decay factor for EMA. Defaults to Config.EMA_DECAY.
            device (str, optional): The device to move the EMA model to.
        """
        self.decay = decay
        # Create a shadow copy of the model
        self.module = copy.deepcopy(model)
        self.module.eval()

        # Move to device if specified, otherwise it stays on the original model's device
        if device:
            self.module.to(device)
            self.device = device
        else:
            self.device = next(self.module.parameters()).device

    def update(self, model):
        """
        Update the EMA model parameters based on the current model.

        Args:
            model (torch.nn.Module): The current training model.
        """
        with torch.no_grad():
            # Update parameters: new_ema = decay * old_ema + (1 - decay) * new_param
            for ema_v, model_v in zip(self.module.parameters(), model.parameters()):
                ema_v.data.mul_(self.decay).add_(model_v.data, alpha=1 - self.decay)

            # Update buffers (e.g., BatchNorm running mean/var) by copying them directly
            for ema_v, model_v in zip(self.module.buffers(), model.buffers()):
                ema_v.copy_(model_v)

    def to(self, device):
        """Moves the EMA model to the specified device."""
        self.module.to(device)
        self.device = device
        return self
