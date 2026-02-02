import os
import random
import numpy as np
import torch
from collections import defaultdict
from library.config import SEED


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to SEED from config.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MetricMonitor:
    """
    A helper class to track and average metrics (e.g., Loss, AUC) during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the internal metric storage."""
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        """
        Updates the running average for a specific metric.

        Args:
            metric_name (str): Name of the metric (e.g., 'Loss').
            val (float): The value to add.
        """
        metric = self.metrics[metric_name]
        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        """
        Returns a string representation of the metrics with full precision.
        """
        return " | ".join([f"{k}: {v['avg']}" for k, v in self.metrics.items()])


def save_checkpoint(model, optimizer, epoch, score, path):
    """
    Saves the model and optimizer state to a checkpoint file.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int): The current epoch.
        score (float): The validation score (e.g., AUC).
        path (str): The file path to save the checkpoint.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "epoch": epoch,
        "score": score,
    }
    torch.save(state, path)


def load_checkpoint(model, optimizer, path, device="cpu"):
    """
    Loads the model and optimizer state from a checkpoint file.

    Args:
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer): The optimizer to load state into (can be None).
        path (str): The file path of the checkpoint.
        device (str): The device to map the location to.

    Returns:
        tuple: (epoch, score) from the saved checkpoint. Returns (0, -inf) if file not found.
    """
    if not os.path.exists(path):
        return 0, -float("inf")

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    score = checkpoint.get("score", 0.0)

    return epoch, score
