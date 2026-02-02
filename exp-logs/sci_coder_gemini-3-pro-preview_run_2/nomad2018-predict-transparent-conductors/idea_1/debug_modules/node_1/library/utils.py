import torch
import numpy as np
import random
import os


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class Normalizer(object):
    """
    Normalize a Tensor and restore it later.
    Useful for standardizing target variables (mean 0, std 1).
    """

    def __init__(self, tensor=None, mean=None, std=None):
        """
        Initialize the Normalizer.

        Args:
            tensor (torch.Tensor, optional): A sample tensor to calculate mean and std from.
            mean (torch.Tensor, optional): Pre-calculated mean.
            std (torch.Tensor, optional): Pre-calculated std.
        """
        if mean is not None and std is not None:
            self.mean = mean
            self.std = std
        elif tensor is not None:
            self.mean = torch.mean(tensor, dim=0)
            self.std = torch.std(tensor, dim=0)
            # Handle zero standard deviation to avoid division by zero
            self.std[self.std == 0] = 1.0
        else:
            self.mean = None
            self.std = None

    def norm(self, tensor):
        """
        Normalize the input tensor.
        """
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer has not been initialized with data.")
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        """
        Inverse normalize the input tensor (restore original scale).
        """
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer has not been initialized with data.")
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        """
        Return the state of the normalizer.
        """
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict):
        """
        Load the state of the normalizer.
        """
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]


class AverageMeter(object):
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
