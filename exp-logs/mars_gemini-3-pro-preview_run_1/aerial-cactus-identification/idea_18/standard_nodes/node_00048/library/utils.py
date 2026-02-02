import os
import random
import numpy as np
import torch
import logging
import sys
from collections import defaultdict
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic operations for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name="cactus_logger"):
    """
    Creates and configures a logger that writes to both stdout and a log file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers if get_logger is called multiple times
    if not logger.handlers:
        # Create formatters
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # Stream Handler (Stdout)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # File Handler
        log_path = os.path.join(Config.WORKING_DIR, "train.log")
        # Ensure directory exists (though setup_directories usually handles this)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class MetricMonitor:
    """
    A utility class to track metrics (loss, accuracy, etc.) during training.
    Computes a running average.
    """

    def __init__(self, float_precision=6):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0, "sum": 0})

    def update(self, metric_name, val, n=1):
        """
        Update the metric with a new value.

        Args:
            metric_name (str): Name of the metric.
            val (float): The value to record (e.g., batch loss).
            n (int): The number of samples associated with this value (e.g., batch size).
        """
        metric = self.metrics[metric_name]

        metric["val"] = val
        metric["sum"] += val * n
        metric["count"] += n
        metric["avg"] = metric["sum"] / metric["count"]

    def __str__(self):
        """
        Returns a string representation of the current averages of all metrics.
        Uses full precision as requested (or high precision).
        """
        return " | ".join(
            [
                "{}: {:.{}f}".format(name, metric["avg"], self.float_precision)
                for name, metric in self.metrics.items()
            ]
        )

    def get_avg(self, metric_name):
        """Returns the current average for a specific metric."""
        return self.metrics[metric_name]["avg"]


def save_checkpoint(model, optimizer, scheduler, epoch, score, path):
    """
    Saves the model checkpoint.

    Args:
        model: The PyTorch model.
        optimizer: The optimizer.
        scheduler: The learning rate scheduler (can be None).
        epoch (int): Current epoch.
        score (float): Validation score (e.g., AUC).
        path (str): Path to save the checkpoint.
    """
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "score": score,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(model, optimizer, scheduler, path, device):
    """
    Loads a model checkpoint.

    Args:
        model: The PyTorch model instance to load weights into.
        optimizer: The optimizer instance to load state into.
        scheduler: The scheduler instance (can be None).
        path (str): Path to the checkpoint file.
        device (str): Device to map the storage to.

    Returns:
        start_epoch (int): The epoch to resume from.
        score (float): The score at the saved epoch.
    """
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if (
        scheduler is not None
        and "scheduler_state_dict" in checkpoint
        and checkpoint["scheduler_state_dict"] is not None
    ):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint.get("epoch", 0)
    score = checkpoint.get("score", 0.0)

    return start_epoch, score
