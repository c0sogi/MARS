import os
import random
import numpy as np
import torch
from collections import defaultdict


def set_seed(seed=42):
    """
    Sets the seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class MetricMonitor:
    """
    A class to track and average metrics (like Loss) during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """
        Resets the internal storage of metrics.
        """
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val, n=1):
        """
        Updates the metric 'metric_name' with value 'val' and weight 'n'.
        """
        metric = self.metrics[metric_name]
        metric["val"] += val * n
        metric["count"] += n
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        """
        Returns a string representation of the averaged metrics.
        Prints full precision as requested.
        """
        return " | ".join(
            [
                f"{metric_name}: {metric['avg']}"
                for metric_name, metric in self.metrics.items()
            ]
        )


def save_checkpoint(model, optimizer, epoch, score, filename):
    """
    Saves the model and optimizer state to a file.
    """
    # Ensure directory exists
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "score": score,
    }
    torch.save(state, filename)


def load_checkpoint(model, filename, optimizer=None, device="cpu"):
    """
    Loads the model and optimizer state from a file.
    Returns the full checkpoint dictionary (containing epoch and score).
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint file not found: {filename}")

    checkpoint = torch.load(filename, map_location=device)

    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer state if provided and present in checkpoint
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
