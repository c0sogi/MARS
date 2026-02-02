import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.seed):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MetricMonitor:
    """
    A utility class to track and average metrics (loss, accuracy, AUC, etc.)
    over an epoch or evaluation phase.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the internal metric storage."""
        self.metrics = {}

    def update(self, metric_name, val, n=1):
        """
        Update the metric with a new value.

        Args:
            metric_name (str): Name of the metric.
            val (float): The value to add (assumed to be an average if n > 1).
            n (int): The number of samples associated with this value.
        """
        val = float(val)
        if metric_name not in self.metrics:
            self.metrics[metric_name] = {"sum": 0.0, "count": 0}
        self.metrics[metric_name]["sum"] += val * n
        self.metrics[metric_name]["count"] += n

    def get_avg(self, metric_name):
        """
        Returns the average value of the requested metric.
        """
        if metric_name not in self.metrics:
            return 0.0
        return self.metrics[metric_name]["sum"] / self.metrics[metric_name]["count"]

    def __str__(self):
        """
        Returns a string representation of the metrics with full precision.
        Format: "metric1: value | metric2: value"
        """
        metric_strs = []
        for name, data in self.metrics.items():
            if data["count"] > 0:
                avg = data["sum"] / data["count"]
                # Print full precision without formatting
                metric_strs.append(f"{name}: {avg}")
        return " | ".join(metric_strs)


def save_checkpoint(model, optimizer, epoch, metrics, filename):
    """
    Saves the model state, optimizer state, and current metrics to a file.
    Ensures the destination directory exists before saving.

    Args:
        model: The PyTorch model.
        optimizer: The PyTorch optimizer.
        epoch (int): Current epoch number.
        metrics (dict): Dictionary of current metrics (e.g., best AUC).
        filename (str): Path to save the checkpoint.
    """
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "epoch": epoch,
        "metrics": metrics,
    }
    torch.save(state, filename)


def load_checkpoint(model, optimizer, filename, device=Config.device):
    """
    Loads the model state and optimizer state from a file.

    Args:
        model: The PyTorch model to load weights into.
        optimizer: The PyTorch optimizer to load state into (can be None).
        filename (str): Path to the checkpoint file.
        device: The device to map the location to.

    Returns:
        tuple: (epoch, metrics) from the checkpoint.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint file not found: {filename}")

    checkpoint = torch.load(filename, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint.get("epoch", 0), checkpoint.get("metrics", {})
