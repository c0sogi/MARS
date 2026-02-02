import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=42):
    """
    Sets the seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class MetricMonitor:
    """
    A utility class to track and average metrics (loss, accuracy, etc.)
    over an epoch.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Clears the metrics."""
        self.metrics = {}

    def update(self, metric_name, val, n=1):
        """
        Update the metric tracking.

        Args:
            metric_name (str): Name of the metric (e.g., 'loss', 'accuracy').
            val (float): The value to add.
            n (int): The number of samples this value represents (usually batch size).
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = {"sum": 0, "count": 0, "avg": 0}

        self.metrics[metric_name]["sum"] += val * n
        self.metrics[metric_name]["count"] += n
        self.metrics[metric_name]["avg"] = (
            self.metrics[metric_name]["sum"] / self.metrics[metric_name]["count"]
        )

    def __str__(self):
        """
        Returns a string representation of the averages of all tracked metrics.
        Prints full precision without rounding.
        """
        return " | ".join(
            [
                f"{metric_name}: {metric_data['avg']}"
                for metric_name, metric_data in self.metrics.items()
            ]
        )

    def get(self, metric_name):
        """Returns the current average of a specific metric."""
        return self.metrics.get(metric_name, {}).get("avg", 0.0)


def save_checkpoint(
    model, optimizer, scheduler, epoch, score, filename="checkpoint.pth"
):
    """
    Saves the model checkpoint to the configured output directory.

    Args:
        model: The PyTorch model.
        optimizer: The optimizer.
        scheduler: The learning rate scheduler.
        epoch (int): Current epoch.
        score (float): Validation score (e.g., accuracy).
        filename (str): Name of the file to save.
    """
    save_path = os.path.join(Config.OUTPUT_DIR, filename)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "score": score,
    }

    torch.save(state, save_path)


def load_checkpoint(model, optimizer=None, scheduler=None, filename="checkpoint.pth"):
    """
    Loads a model checkpoint.

    Args:
        model: The PyTorch model instance.
        optimizer: The optimizer instance (optional).
        scheduler: The scheduler instance (optional).
        filename (str): The filename or full path to load from.

    Returns:
        start_epoch (int): The epoch to resume from.
        score (float): The best score recorded in the checkpoint.
    """
    # Determine if filename is a path or just a name in OUTPUT_DIR
    if os.path.exists(filename):
        load_path = filename
    else:
        load_path = os.path.join(Config.OUTPUT_DIR, filename)

    if not os.path.exists(load_path):
        return 0, 0.0

    # Load to CPU first to avoid GPU OOM during loading
    checkpoint = torch.load(load_path, map_location=torch.device("cpu"))

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and checkpoint["optimizer_state_dict"]:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and checkpoint["scheduler_state_dict"]:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint.get("epoch", 0)
    score = checkpoint.get("score", 0.0)

    return start_epoch, score
