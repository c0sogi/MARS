import os
import random
import copy
import numpy as np
import torch
from library.config import SEED


def set_seed(seed=SEED):
    """
    Sets the random seed for python, numpy, and torch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to SEED from config.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def count_parameters(model):
    """
    Counts the total number of trainable parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): The model to inspect.

    Returns:
        int: The number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ModelEMA:
    """
    Maintains a moving average of model parameters using Exponential Moving Average (EMA).
    This improves model generalization and stability.
    """

    def __init__(self, model, decay=0.999):
        """
        Args:
            model (torch.nn.Module): The model to track.
            decay (float): The decay factor. Defaults to 0.999.
        """
        self.decay = decay
        # Create a deep copy of the model for the EMA weights
        self.ema = copy.deepcopy(model)
        self.ema.eval()

        # Disable gradients for the shadow model to save memory/compute
        for param in self.ema.parameters():
            param.requires_grad = False

    def update(self, model):
        """
        Update the shadow parameters based on the current model parameters.

        Args:
            model (torch.nn.Module): The current training model.
        """
        with torch.no_grad():
            # State dict includes both parameters and persistent buffers (e.g. running_mean)
            msd = model.state_dict()
            esd = self.ema.state_dict()

            for name, m_tensor in msd.items():
                if name in esd:
                    e_tensor = esd[name]

                    # Update floating point tensors (weights, biases, running stats)
                    if m_tensor.dtype.is_floating_point:
                        e_tensor.mul_(self.decay).add_(m_tensor, alpha=1.0 - self.decay)
                    else:
                        # Directly copy non-floating point tensors (e.g. num_batches_tracked)
                        e_tensor.copy_(m_tensor)

    def get_model(self):
        """
        Returns the shadow model (EMA model).
        """
        return self.ema
