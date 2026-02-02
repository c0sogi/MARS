import os
import random
import numpy as np
import torch
import logging
import sys
from collections import defaultdict

# Import Config to ensure consistency with the rest of the pipeline
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name="training"):
    """
    Creates and configures a logger that prints to stdout.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Check if handlers already exist to avoid duplicate logs
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


class MetricMonitor:
    """
    A utility class to track and compute the average of metrics (e.g., Loss, Accuracy)
    during training or validation loops.
    """

    def __init__(self, float_precision=4):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val, n=1):
        """
        Update the metric with a new value.

        Args:
            metric_name (str): Name of the metric.
            val (float): The value to add.
            n (int): The number of samples this value represents (usually batch size).
        """
        metric = self.metrics[metric_name]
        metric["val"] += val * n
        metric["count"] += n
        metric["avg"] = metric["val"] / metric["count"]

    def get_avg(self, metric_name):
        """
        Returns the current average of the specified metric.
        """
        return self.metrics[metric_name]["avg"]

    def __str__(self):
        """
        Returns a string representation of the current averages for all tracked metrics.
        """
        return " | ".join(
            [
                "{}: {:.{prec}f}".format(
                    metric_name, metric["avg"], prec=self.float_precision
                )
                for metric_name, metric in self.metrics.items()
            ]
        )


def save_checkpoint(state, filename):
    """
    Saves the model state (and optimizer/scheduler state) to a file.

    Args:
        state (dict): The state dictionary containing model weights, optimizer state, etc.
        filename (str): The path where the checkpoint will be saved.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)


def load_checkpoint(
    filename, model, optimizer=None, scheduler=None, device=Config.DEVICE
):
    """
    Loads a checkpoint into the model (and optionally optimizer/scheduler).

    Args:
        filename (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The scheduler to load state into.
        device (str): Device to map the location to.

    Returns:
        dict: The raw checkpoint dictionary (in case extra info is needed like epoch or best_score).
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint file not found: {filename}")

    checkpoint = torch.load(filename, map_location=device)

    # Handle state_dict key if present, otherwise assume the dict itself is the state_dict
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint
