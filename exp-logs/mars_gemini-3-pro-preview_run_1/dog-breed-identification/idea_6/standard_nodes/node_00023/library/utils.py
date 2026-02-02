import os
import torch
from collections import defaultdict
from library.config import Config, seed_everything


class MetricMonitor:
    """
    A utility class to track and aggregate metrics (e.g., loss, accuracy)
    over an epoch or a set of batches.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all tracked metrics to zero."""
        self.metrics = defaultdict(lambda: {"sum": 0.0, "count": 0, "avg": 0.0})

    def update(self, metric_name, val, n=1):
        """
        Updates the specified metric with a new value.

        Args:
            metric_name (str): The name of the metric to update.
            val (float): The value to record (e.g., batch loss).
            n (int): The number of samples corresponding to this value (default: 1).
        """
        metric = self.metrics[metric_name]
        metric["sum"] += val * n
        metric["count"] += n
        metric["avg"] = metric["sum"] / metric["count"]

    def get_avg(self, metric_name):
        """Returns the current average for the specified metric."""
        return self.metrics[metric_name]["avg"]

    def get_all_avgs(self):
        """Returns a dictionary mapping metric names to their current averages."""
        return {k: v["avg"] for k, v in self.metrics.items()}

    def __str__(self):
        """
        Returns a string representation of the metrics.
        Note: The training loop is responsible for printing full precision if needed.
        """
        return " | ".join([f"{k}: {v['avg']:.4f}" for k, v in self.metrics.items()])


def get_checkpoint_path(filename):
    """
    Constructs the full path for a checkpoint file within the configured working directory.
    Ensures the directory exists.

    Args:
        filename (str): The name of the checkpoint file.

    Returns:
        str: The full absolute or relative path to the file.
    """
    os.makedirs(Config.working_dir, exist_ok=True)
    return os.path.join(Config.working_dir, filename)


def save_checkpoint(state, filename="checkpoint.pth"):
    """
    Saves a model checkpoint (state dict) to the configured working directory.

    Args:
        state (dict): The dictionary containing model state, optimizer state, etc.
        filename (str): The name of the file to save.
    """
    file_path = get_checkpoint_path(filename)
    torch.save(state, file_path)
