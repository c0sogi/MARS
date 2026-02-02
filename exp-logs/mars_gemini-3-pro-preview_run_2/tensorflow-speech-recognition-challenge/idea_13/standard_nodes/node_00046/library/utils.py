import os
import random
import numpy as np
import torch
from copy import deepcopy


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class MetricMonitor:
    """
    A utility class to track and compute the running average of metrics (e.g., loss, accuracy).
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the internal state of the monitor."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates the metric with a new value.

        Args:
            val (float): The value to add.
            n (int): The weight of the value (usually batch size).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def result(self):
        """Returns the current average."""
        return self.avg


class ModelEMA:
    """
    Implements Exponential Moving Average (EMA) for model weights.
    Maintains a shadow copy of the model that is updated with a decay factor.
    This helps stabilize training and often improves generalization.
    """

    def __init__(self, model, decay=0.999, device=None):
        """
        Args:
            model (torch.nn.Module): The model to track.
            decay (float): The decay factor for EMA (default: 0.999).
            device (torch.device, optional): The device to store the shadow model on.
                                             If None, uses the same device as the input model.
        """
        self.decay = decay
        self.model = model
        self.shadow = deepcopy(self.model)
        self.shadow.eval()
        self.device = device

        if self.device is not None:
            self.shadow.to(self.device)

        # Disable gradients for the shadow model to save memory/computation
        for param in self.shadow.parameters():
            param.requires_grad = False

    def update(self, model):
        """
        Updates the shadow model parameters using the current model parameters.

        Args:
            model (torch.nn.Module): The current training model.
        """
        with torch.no_grad():
            msd = model.state_dict()
            # Iterate through the shadow model's state dict
            for k, v in self.shadow.state_dict().items():
                if k in msd:
                    model_v = msd[k].detach()

                    if self.device is not None:
                        model_v = model_v.to(self.device)

                    # Update floating point parameters/buffers with EMA
                    if v.dtype.is_floating_point:
                        v.copy_(v * self.decay + (1.0 - self.decay) * model_v)
                    # Directly copy integer buffers (e.g., num_batches_tracked in BatchNorm)
                    else:
                        v.copy_(model_v)

    def module(self):
        """
        Returns the shadow model (the EMA model).
        """
        return self.shadow
