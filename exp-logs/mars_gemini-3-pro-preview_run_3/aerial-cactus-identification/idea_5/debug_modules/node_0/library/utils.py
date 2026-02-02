import os
import random
import copy
import numpy as np
import torch
from collections import defaultdict
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    Also configures cuDNN for deterministic behavior where possible, while keeping
    benchmark enabled for performance on fixed-size inputs.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Enable benchmark for optimization on fixed input sizes (32x32)
    # Enable deterministic algorithms for reproducibility
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def save_checkpoint(model, path):
    """
    Saves the model state dictionary to the specified path.
    Uses copy.deepcopy to ensure the saved state is immutable and not
    affected by subsequent optimizer steps.

    Args:
        model (torch.nn.Module): The model to save.
        path (str): The file path to save the checkpoint to.
    """
    # Ensure the directory exists
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Deep copy the state dict to ensure immutability
    state_dict = copy.deepcopy(model.state_dict())
    torch.save(state_dict, path)


class MetricMonitor:
    """
    A utility class to track and average metrics (loss, accuracy, etc.)
    over a sequence of steps.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the internal metrics storage."""
        self.metrics = defaultdict(lambda: {"sum": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        """
        Updates the metric with a new value.

        Args:
            metric_name (str): Name of the metric (e.g., 'loss').
            val (float or torch.Tensor): The value to add.
        """
        metric = self.metrics[metric_name]

        # Detach and convert tensor to float if necessary
        if isinstance(val, torch.Tensor):
            val = val.detach().cpu().item()

        metric["sum"] += val
        metric["count"] += 1
        metric["avg"] = metric["sum"] / metric["count"]

    def get_avg(self, metric_name):
        """Returns the current average for the specified metric."""
        return self.metrics[metric_name]["avg"]

    def __str__(self):
        """
        Returns a string representation of the current averages for all tracked metrics.
        Prints full precision without rounding.
        """
        metrics_str = []
        for name, data in self.metrics.items():
            # Using str() ensures full precision is printed
            metrics_str.append(f"{name}: {data['avg']}")
        return " | ".join(metrics_str)
