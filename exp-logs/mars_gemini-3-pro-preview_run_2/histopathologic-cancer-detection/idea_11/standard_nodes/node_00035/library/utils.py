import os
import random
import numpy as np
import torch
import torch.nn as nn
import logging
import copy
from collections import defaultdict
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # benchmark=False ensures deterministic algorithm selection
    torch.backends.cudnn.benchmark = False


def setup_logger(log_file, level=logging.INFO):
    """
    Sets up a logger to write to both console and a file.

    Args:
        log_file (str): Path to the log file.
        level (int): Logging level (e.g., logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create directory for log file if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("training_logger")
    logger.setLevel(level)

    # Avoid adding handlers multiple times if logger is already configured
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # File Handler
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Stream Handler (Console)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


class MetricMonitor:
    """
    A helper class to track and average metrics (loss, accuracy, etc.) over an epoch.
    """

    def __init__(self, float_precision=6):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        """Resets the internal state."""
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        """
        Updates the metric with a new value.

        Args:
            metric_name (str): Name of the metric.
            val (float): Value to add.
        """
        metric = self.metrics[metric_name]
        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        """
        Returns a formatted string of the current average metrics.
        """
        return " | ".join(
            [
                "{}: {:.{prec}f}".format(
                    metric_name, metric["avg"], prec=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )


class ModelEma(nn.Module):
    """
    Implements Exponential Moving Average (EMA) for model weights.
    Crucial for the Memory-Resident Homogeneous Bagged Ensemble strategy.
    """

    def __init__(self, model, decay=0.9999, device=None):
        super(ModelEma, self).__init__()
        # Create a deep copy of the model to serve as the shadow model
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device

        # Move shadow model to the specified device
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(
                self.module.state_dict().values(), model.state_dict().values()
            ):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        """
        Update the EMA model parameters based on the current model parameters.
        Formula: ema_param = decay * ema_param + (1 - decay) * current_param
        """
        self._update(
            model, update_fn=lambda e, m: self.decay * e + (1.0 - self.decay) * m
        )

    def set(self, model):
        """
        Set the EMA model parameters to exactly match the current model parameters.
        """
        self._update(model, update_fn=lambda e, m: m)
