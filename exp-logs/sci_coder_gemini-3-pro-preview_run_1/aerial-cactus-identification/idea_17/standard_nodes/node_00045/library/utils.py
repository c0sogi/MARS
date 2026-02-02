import os
import sys
import random
import numpy as np
import torch
import logging
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the seed for generating random numbers for full reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name=__name__):
    """
    Creates and configures a logger that writes to both stdout and a file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Check if handlers are already added to avoid duplicate logs
    if not logger.handlers:
        # Create handlers
        c_handler = logging.StreamHandler(sys.stdout)

        log_file = os.path.join(Config.WORKING_DIR, "train.log")
        f_handler = logging.FileHandler(log_file)

        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(logging.INFO)

        # Create formatters and add it to handlers
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        c_handler.setFormatter(formatter)
        f_handler.setFormatter(formatter)

        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

    return logger


class MetricMonitor:
    """
    A utility class to track running averages of metrics (loss, accuracy, etc.).
    """

    def __init__(self, float_precision=4):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = {}

    def update(self, metric_name, val):
        """
        Update the running average for a specific metric.

        Args:
            metric_name (str): Name of the metric.
            val (float): Value to update.
        """
        previous_data = self.metrics.get(metric_name, {"count": 0, "sum": 0})
        previous_data["count"] += 1
        previous_data["sum"] += val
        self.metrics[metric_name] = previous_data

    def get_avg(self, metric_name):
        """Returns the average value of a specific metric."""
        data = self.metrics.get(metric_name, {"count": 0, "sum": 0})
        if data["count"] == 0:
            return 0
        return data["sum"] / data["count"]

    def __str__(self):
        """
        Returns a string representation of the current averages of all metrics.
        """
        return " | ".join(
            [
                "{}: {:.{prec}f}".format(
                    metric_name, self.get_avg(metric_name), prec=self.float_precision
                )
                for metric_name in sorted(self.metrics.keys())
            ]
        )


def sigmoid(x):
    """
    Computes the sigmoid function.
    """
    return 1 / (1 + np.exp(-x))


def softmax(x):
    """
    Computes the softmax function.
    """
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)
